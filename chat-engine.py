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
        "deepdive": "Detailed content",
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
            model_kwargs={
                "max_tokens": 500,
                "temperature": 0.25,
                "top_p": 0.85,
                "top_k": 40,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.0,
                "num_generations": 1,
            },
        )
        self.embed = OCIGenAIEmbeddings(
            model_id=props.getEmbeddingModelName(),
            service_endpoint=props.getEndpoint(),
            compartment_id=props.getCompartment(),
        )
        self.db = FAISS.load_local(index_dir, self.embed, allow_dangerous_deserialization=True)

        self.custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful AI assistant. Using the context below, answer the question in a clear, "
                "professional and slightly more elaborate way with HTML formatting.\n\n"
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            ),
        )

    def classify_with_genai(self, query: str) -> Tuple[str | None, str | None]:
        """
        Classify the user query into (audience, type) labels using OCI Generative AI.
        """
        audience_expl = "\n".join(f"- {k}: {v}" for k, v in self.ALLOWED_AUDIENCES.items())
        type_expl = "\n".join(f"- {k}: {v}" for k, v in self.ALLOWED_TYPES.items())
        extra_rules = (
            "Guidelines:\n"
            "• Choose *internal* only if the question explicitly references Oracle internal docs, processes or tools.\n"
            "• Choose *governance* only for policy, compliance or review-process queries.\n"
            "Examples:\n"
            "  Q: 'What does the Oracle AI policy say about model training data?' → internal_governance\n"
            "  Q: 'Explain RAG architecture in simple terms.' → general_insight\n"
            "  Q: 'Show me the Python SDK for OCI Generative AI.' → technical_deepdive\n"
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

    def ranked_retrieval(self, query: str, audience: str, doc_type: str, k: int = 5) -> List:
        """
        Retrieve up to k docs, prioritizing metadata matches for audience and type.
        """
        results: List[Tuple[int, object]] = []

        aud_rank = [audience] + [a for a in self.ALLOWED_AUDIENCES if a != audience]
        typ_rank = [doc_type] + [t for t in self.ALLOWED_TYPES if t != doc_type]

        for a_idx, aud in enumerate(aud_rank):
            for t_idx, typ in enumerate(typ_rank):
                filter_dict = {"audience": aud, "type": typ}
                retr = self.db.as_retriever(
                    search_type="similarity", search_kwargs={"k": k, "filter": filter_dict}
                )
                for d in retr.invoke(query):
                    results.append((a_idx + t_idx, d))

        best: OrderedDict[str, Tuple[int, object]] = OrderedDict()
        for score, doc in results:
            key = doc.metadata.get("source", "") + "#" + str(doc.metadata.get("page", 0))
            if key not in best or score < best[key][0]:
                best[key] = (score, doc)

        ordered = sorted(best.values(), key=lambda x: x[0])[:k]
        return [d for _score, d in ordered]

    def chat_stream(self, message: str):
        """
        Stream HTML-formatted answer chunks and source list for Gradio interface.
        """
        if self.debug:
            print("\n[DEBUG] New user question:", message)

        if message.lower().strip() == "reset session":
            yield "Session reset. Ask away!"
            return

        audience, doc_type = self.classify_with_genai(message)
        if not audience or not doc_type:
            yield self.generate_clarifying_prompt(message)
            return

        if self.debug:
            print(f"[DEBUG] Classified as {audience}_{doc_type}")

        docs = self.ranked_retrieval(message, audience, doc_type)
        if docs:
            if self.debug:
                print(f"[DEBUG] Filtered doc list ({len(docs)} docs):")
                for i, d in enumerate(docs):
                    meta = d.metadata or {}
                    print(
                        f"    • {i}: {Path(meta.get('source', 'Unknown')).name} | "
                        f"aud={meta.get('audience')} | type={meta.get('type')}"
                    )
        else:
            if self.debug:
                print("[DEBUG] No docs found, performing fallback search")
            docs = list(self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5}).invoke(message))
            if self.debug:
                print(f"[DEBUG] Fallback doc list ({len(docs)} docs):")
                for i, d in enumerate(docs):
                    meta = d.metadata or {}
                    print(
                        f"    • {i}: {Path(meta.get('source', 'Unknown')).name} | "
                        f"aud={meta.get('audience')} | type={meta.get('type')}"
                    )

        context_text = "\n\n".join(d.page_content for d in docs)
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
            link = f'<a href="file://{Path(src).resolve()}" target="_blank">{fn}</a>'
            items.append(f"<li>{link} ({meta.get('audience', 'unknown')} | {meta.get('type', 'unknown')})</li>")

        if items:
            sources_html = (
                '<div style="font-size:13px;margin-top:15px;color:#444;">'
                '<strong>Sources used:</strong><ul>' + "".join(items) + '</ul></div>'
            )
            yield partial.replace("\n", "<br>") + sources_html

    def chat(self, message: str) -> Tuple[str, List]:
        """
        Perform a one-shot chat returning raw text and source documents for CLI use.
        """
        if self.debug:
            print("\n[DEBUG] New user question:", message)

        if message.lower().strip() == "reset session":
            return "Session reset. Ask away!", []

        audience, doc_type = self.classify_with_genai(message)
        if not audience or not doc_type:
            return self.generate_clarifying_prompt(message), []

        docs = self.ranked_retrieval(message, audience, doc_type)
        context_text = "\n\n".join(d.page_content for d in docs)
        answer_prompt = self.custom_prompt.format(context=context_text, question=message)

        if hasattr(self.llm, "stream"):
            response = "".join(chunk.content for chunk in self.llm.stream([HumanMessage(content=answer_prompt)]))
        else:
            response = self.llm.invoke([HumanMessage(content=answer_prompt)]).content

        return response, docs