#!/usr/bin/env python3
"""
Common chat engine for RAG-enabled OCI Generative AI applications.
Includes classification, retrieval, and streaming/chat logic shared by chatpion-web.py and chatpion-cli.py.
"""
import re
import os
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores import FAISS

import importlib.util, os

# dynamic load of configuration loader (load-config.py)
spec_cfg = importlib.util.spec_from_file_location(
    "load_config", os.path.join(os.path.dirname(__file__), "load-config.py")
)
cfg_mod = importlib.util.module_from_spec(spec_cfg)
spec_cfg.loader.exec_module(cfg_mod)
LoadConfig = cfg_mod.LoadConfig


class ChatEngine:
    """
    ChatEngine encapsulates classification, retrieval, and chat operations
    for RAG-enabled OCI Generative AI assistants.
    """

    ALLOWED_AUDIENCES = {
        "business": "Executives, sales, strategy, partners, human resources, people, society…",
        "technical": "Developers, architects, engineers, quality assurance, devops, system design…",
        "general": "Non‑technical and non‑business content intended for a broad audience…",
        "internal": "Oracle internal documents. Assume the user is an Oracle employee.",
    }

    ALLOWED_TYPES = {
        "insight": "Conceptual, strategic thinking or high‑level concepts",
        "deepdive": "Detailed content, excluding internal procedures or standards.",
        "research": "Research‑focused publications",
        "governance": "Internal tools, corporate guidelines, policies and processes",
    }

    def __init__(self, debug: bool = False, index_dir: str = "faiss_index"):
        self.debug = debug
        props = LoadConfig()

        self.llm = ChatOCIGenAI(
            model_id=props.getModelName(),
            service_endpoint=props.getEndpoint(),
            compartment_id=props.getCompartment(),
            model_kwargs = {
                "max_tokens": 800,               # LLaMA 3 handles long outputs well. 800 gives more room.
                "temperature": 0.2,              # Lower = more deterministic. Ideal for factual answers.
                "top_p": 0.9,                    # Keeps diversity without harming coherence.
                "top_k": 40,                     # Helps filter low-probability noise, but optional.
                "frequency_penalty": 0.1,        # Light penalty to avoid repeated phrases.
                "presence_penalty": 0.0,         # Neutral. Don't push novelty.
                "num_generations": 1,           # Only one needed for assistant behaviour.
            }
        )
        self.embed = OCIGenAIEmbeddings(
            model_id=props.getEmbeddingModelName(),
            service_endpoint=props.getEndpoint(),
            compartment_id=props.getCompartment(),
        )
        self.db = FAISS.load_local(index_dir, self.embed, allow_dangerous_deserialization=True)

        if self.debug:
            print(f"[DEBUG] FAISS index contains {len(self.db.docstore._dict)} docs")
            count = sum(
                1 for d in self.db.docstore._dict.values()
                if d.metadata.get("audience") == "internal"
                and d.metadata.get("type") == "governance"
                and d.metadata.get("oracle_owned") is True
            )
            print(f"[DEBUG] internal-governance oracle_owned=True docs: {count}")

        self.custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful AI assistant. Using the context below, answer the question in a clear, "
                "professional and slightly more elaborate way with HTML formatting.\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            ),
        )
        # history of (user question, assistant answer) for multi-turn context
        self.session_history: List[Tuple[str, str]] = []

    def rephrase_query(self, question: str) -> str:
        """
        Use LLM to rephrase the user's question to improve retrieval quality.
        """
        prompt = (
            "You're an AI assistant helping improve search accuracy.\n"
            "Rephrase or expand the following user question so it's clearer and more likely to retrieve relevant internal Oracle documents.\n"
            "Avoid changing the meaning, just make it clearer or more formal.\n\n"
            f"Original Question:\n{question}\n\nImproved Question:"
        )
        if self.debug:
            print(f"[DEBUG] Rephrasing prompt:\n{prompt}")
        return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

    def classify_with_genai(self, query: str) -> Tuple[str | None, str | None]:
        """
        Classify the user query into (audience, type) labels using OCI Generative AI.
        """
        audience_expl = "\n".join(f"- {k}: {v}" for k, v in self.ALLOWED_AUDIENCES.items())
        type_expl = "\n".join(f"- {k}: {v}" for k, v in self.ALLOWED_TYPES.items())
        extra_rules = (
            "Guidelines:\n"
            "• Always assume the user is an Oracle employee. Questions may refer to internal Oracle tools, processes, or documentation.\n"
            "• Default to *internal* unless the question clearly applies beyond Oracle, with high confidence (e.g. public tech concepts, general knowledge).\n"
            "• Choose *governance* only for policy, compliance, or review-process queries.\n"
            "Examples:\n"
            "  Q: 'What tools can I use?' → internal_governance\n"
            "  Q: 'Can I use tool <any tool name>?' → internal_governance\n"
            "  Q: 'What does the policy say about model training data?' → internal_governance\n"
            "  Q: 'What is the process to get approval for using external third party tools?' → internal_governance\n"
            "  Q: 'Explain RAG architecture in simple terms.' → general_insight\n"
            "  Q: 'Show me the Python SDK for OCI Generative AI.' → technical_deepdive\n"
            "  Q: 'How do I request access to the internal fine-tuning service?' → internal\n"
        )

        prompt = (
            "You are an AI classification assistant. Classify the user question into two labels (audience and type).\n\n"
            f"Valid Audience values:\n{audience_expl}\n\n"
            f"Valid Type values:\n{type_expl}\n\n"
            f"{extra_rules}\n"
            "Return the labels in the exact format `audience_type` with no other text.\n\n"
            f"User question:\n{query}\n\nYour response:"
        )
        if self.debug:
            print("\n[DEBUG] Classification prompt:\n", prompt)

        raw = (
            self.llm.invoke([HumanMessage(content=prompt)])
            .content.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if self.debug:
            print("[DEBUG] Classification response:", raw)

        pattern = rf"^({'|'.join(self.ALLOWED_AUDIENCES)})_({'|'.join(self.ALLOWED_TYPES)})$"
        m = re.match(pattern, raw)
        return (m.group(1), m.group(2)) if m else (None, None)

    def generate_clarifying_prompt(self, msg: str) -> str:
        """
        Generate a concise follow-up question when classification confidence is low.
        """
        prompt = (
            "You're an AI assistant that could not confidently classify the user's intent.\n"
            "Ask one concise follow-up question to clarify both audience and type.\n\n"
            f"User message:\n{msg}\n"
        )
        if self.debug:
            print("[DEBUG] Clarifying prompt input:\n", prompt)
        return self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

    def ranked_retrieval(
        self, query: str, audience: str, doc_type: str, k: int = 5
    ) -> List:
        """
        Return up to k docs ranked for relevance.
        • Oracle-owned docs are mandatory for internal queries.
        • Large PDFs (many chunks) are penalised so one file cannot dominate.
        """
        raw: List[Tuple[int, object]] = []

        # iterate audience/type permutations, best matches first
        aud_rank = [audience] + [a for a in self.ALLOWED_AUDIENCES if a != audience]
        typ_rank = [doc_type] + [t for t in self.ALLOWED_TYPES if t != doc_type]

        for a_idx, aud in enumerate(aud_rank):
            for t_idx, typ in enumerate(typ_rank):
                flt = {"audience": aud, "type": typ}
                if self.debug:
                    print(f"[DEBUG] Running filter: {flt}")
                docs = self.db.as_retriever(
                    search_type="similarity", search_kwargs={"k": k, "filter": flt}
                ).invoke(query)
                if self.debug:
                    print(f"[DEBUG] Retrieved {len(docs)} docs for {flt}")
                raw.extend((a_idx + t_idx, d) for d in docs)

        # for base_score, doc in raw:
        #     meta = doc.metadata or {}
        #     print(f"[DEBUG] Evaluating doc: {meta.get('source')} | oracle_owned: {meta.get('oracle_owned')}")

        if not raw:
            return []  # let caller decide on fallback

        ranked: OrderedDict[str, Tuple[int, object]] = OrderedDict()
        for base_score, doc in raw:
            meta = doc.metadata or {}

            # internal queries must be Oracle-owned
            if audience == "internal" and not meta.get("oracle_owned"):
                continue

            score = base_score

            # +1 boost for exact audience/type match
            if meta.get("audience") == audience and meta.get("type") == doc_type:
                score -= 1

            # extra boost for Oracle-owned internal docs
            if audience == "internal" and meta.get("oracle_owned"):
                score -= 1

            # PENALTY: add 1 point for every 5 chunks beyond 10
            chunk_count = int(meta.get("chunk_count", 1))
            score += max(0, (chunk_count - 10) // 5)

            key = f"{meta.get('source','unknown')}#{meta.get('page',0)}"
            if key not in ranked or score < ranked[key][0]:
                ranked[key] = (score, doc)

        ordered = sorted(ranked.values(), key=lambda x: x[0])[:k]
        if self.debug:
            print(f"[DEBUG] Final ranked doc count: {len(ordered)}")
            for i, (_, d) in enumerate(ordered):
                m = d.metadata
                print(f"    • {i}: {Path(m.get('source')).name} "
                    f"| aud={m.get('audience')} type={m.get('type')} "
                    f"chunks={m.get('chunk_count')} oracle={m.get('oracle_owned')}")
        return [doc for _, doc in ordered]

    def chat_stream(self, message: str):
        if self.debug:
            print("\n[DEBUG] New user question:", message)

        if message.lower().strip() == "reset session":
            self.session_history.clear()
            yield "Session reset. Ask away!"
            return

        audience, doc_type = self.classify_with_genai(message)
        if not audience or not doc_type:
            yield self.generate_clarifying_prompt(message)
            return

        if self.debug:
            print(f"[DEBUG] Classified as {audience}_{doc_type}")

        # Enhance the query before performing search
        enhanced_query = self.rephrase_query(message)
        if self.debug:
            print(f"[DEBUG] Enhanced query:\n{enhanced_query}")

        docs = self.ranked_retrieval(enhanced_query, audience, doc_type)

        if not docs:
            if audience == "internal":
                yield "No internal Oracle-authored content is available to reliably answer your question."
                return
            if self.debug:
                print("[DEBUG] No docs found, performing fallback search (non-internal)")
            docs = self.db.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            ).invoke(enhanced_query)
            if self.debug:
                print(f"[DEBUG] Fallback doc list ({len(docs)} docs):")
                for i, d in enumerate(docs):
                    meta = d.metadata or {}
                    print(
                        f"    • {i}: {Path(meta.get('source', 'Unknown')).name} | "
                        f"aud={meta.get('audience')} | type={meta.get('type')}"
                    )

        history_text = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in self.session_history)
        doc_context = "\n\n".join(d.page_content for d in docs)
        context_text = f"{history_text}\n\n{doc_context}" if history_text else doc_context
        answer_prompt = self.custom_prompt.format(context=context_text, question=message)

        partial = ""
        if hasattr(self.llm, "stream"):
            for chunk in self.llm.stream([HumanMessage(content=answer_prompt)]):
                partial += chunk.content
                yield partial.replace("\n", "<br>")
        else:
            partial = self.llm.invoke([HumanMessage(content=answer_prompt)]).content
            yield partial.replace("\n", "<br>")

        seen, items = set(), []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("source", "Unknown")
            if src in seen:
                continue
            seen.add(src)
            fn = Path(src).name
            project_root = Path(__file__).parent.resolve()
            try:
                rel_path = Path(src).resolve().relative_to(project_root).as_posix()
                href = f"/gradio_api/file={rel_path}"
            except ValueError:
                href = f"file://{Path(src).resolve()}"
            label = "Oracle" if meta.get("oracle_owned") else "External"
            link = f'<a href="{href}" target="_blank">{fn}</a>'
            items.append(f"<li>{link} ({label} | {meta.get('audience', 'unknown')} | {meta.get('type', 'unknown')})</li>")

        if items:
            sources_html = (
                '<div style="font-size:13px;margin-top:15px;color:#444;">'
                '<strong>Sources used:</strong><ul>' + "".join(items) + '</ul></div>'
            )
            yield partial.replace("\n", "<br>") + sources_html

        self.session_history.append((message, partial))

    def chat(self, message: str) -> Tuple[str, List]:
        if self.debug:
            print("\n[DEBUG] New user question:", message)

        if message.lower().strip() == "reset session":
            self.session_history.clear()
            return "Session reset. Ask away!", []

        audience, doc_type = self.classify_with_genai(message)
        if not audience or not doc_type:
            return self.generate_clarifying_prompt(message), []

        # Enhance the query before performing search
        enhanced_query = self.rephrase_query(message)
        if self.debug:
            print(f"[DEBUG] Enhanced query:\n{enhanced_query}")

        docs = self.ranked_retrieval(enhanced_query, audience, doc_type)

        if not docs and audience == "internal":
            return "No internal Oracle-authored content is available to reliably answer your question.", []
        if not docs:
            docs = list(
                self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5}).invoke(enhanced_query)
            )

        history_text = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in self.session_history)
        doc_context = "\n\n".join(d.page_content for d in docs)
        context_text = f"{history_text}\n\n{doc_context}" if history_text else doc_context
        answer_prompt = self.custom_prompt.format(context=context_text, question=message)

        if hasattr(self.llm, "stream"):
            response = "".join(chunk.content for chunk in self.llm.stream([HumanMessage(content=answer_prompt)]))
        else:
            response = self.llm.invoke([HumanMessage(content=answer_prompt)]).content

        self.session_history.append((message, response))
        return response, docs
  
        """
        Perform a one-shot chat returning raw text and source documents for CLI use.
        """
        if self.debug:
            print("\n[DEBUG] New user question:", message)

        if message.lower().strip() == "reset session":
            self.session_history.clear()
            return "Session reset. Ask away!", []

        audience, doc_type = self.classify_with_genai(message)
        if not audience or not doc_type:
            return self.generate_clarifying_prompt(message), []

        docs = self.ranked_retrieval(message, audience, doc_type)
        if not docs and audience == "internal":
            return "No internal Oracle-authored content is available to reliably answer your question.", []
        if not docs:
            docs = list(
                self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5}).invoke(message)
            )
        
        history_text = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in self.session_history)
        doc_context = "\n\n".join(d.page_content for d in docs)
        context_text = f"{history_text}\n\n{doc_context}" if history_text else doc_context
        answer_prompt = self.custom_prompt.format(context=context_text, question=message)

        if hasattr(self.llm, "stream"):
            response = "".join(chunk.content for chunk in self.llm.stream([HumanMessage(content=answer_prompt)]))
        else:
            response = self.llm.invoke([HumanMessage(content=answer_prompt)]).content

        # update history with the full assistant response
        self.session_history.append((message, response))
        return response, docs