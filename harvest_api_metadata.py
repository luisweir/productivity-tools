#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

# ---------------- Common utils ----------------

def read_json(p: Path) -> Dict[str, Any]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Skipping {p}: {e}", file=sys.stderr)
        return {}

def is_oas2(doc: Dict[str, Any]) -> bool:
    return isinstance(doc, dict) and doc.get("swagger") == "2.0"

def is_oas3(doc: Dict[str, Any]) -> bool:
    if not isinstance(doc, dict):
        return False
    ov = doc.get("openapi")
    return isinstance(ov, str) and ov.startswith("3.0")

def flat_join(items: Any) -> str:
    if items is None:
        return ""
    if isinstance(items, (list, tuple, set)):
        return " | ".join(str(x) for x in items if x is not None)
    return str(items)

def clean(x: Any) -> str:
    if x is None:
        return ""
    return " ".join(str(x).split())

def resolve_ref(doc: Dict[str, Any], ref: str) -> Any:
    if not ref or not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    node: Any = doc
    for part in ref.lstrip("#/").split("/"):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node

def tag_desc_map(doc: Dict[str, Any]) -> Dict[str, str]:
    m = {}
    for t in doc.get("tags", []) or []:
        if isinstance(t, dict) and t.get("name"):
            m[t["name"]] = clean(t.get("description"))
    return m

def security_to_str(sec: Any) -> str:
    if not sec:
        return ""
    parts = []
    for item in sec:
        if isinstance(item, dict):
            for k, v in item.items():
                scopes = flat_join(v)
                parts.append(f"{k}{'(' + scopes + ')' if scopes else ''}")
    return flat_join(parts)

# ---------------- OAS2 extraction ----------------

def get_info_oas2(sw: Dict[str, Any]) -> Dict[str, str]:
    info = sw.get("info", {}) or {}
    return {
        "api_title": clean(info.get("title")),
        "api_version": clean(info.get("version")),
        "api_description": clean(info.get("description")),
        "api_termsOfService": clean(info.get("termsOfService")),
        "api_contact": clean((info.get("contact") or {}).get("email") or (info.get("contact") or {}).get("name")),
        "api_license": clean((info.get("license") or {}).get("name")),
        "host": clean(sw.get("host")),
        "basePath": clean(sw.get("basePath")),
        "schemes": flat_join(sw.get("schemes")),
        "consumes": flat_join(sw.get("consumes")),
        "produces": flat_join(sw.get("produces")),
        "externalDocs": clean(((sw.get("externalDocs") or {}).get("url"))),
        "spec_version": "OAS2",
    }

def collect_parameters_oas2(sw: Dict[str, Any], path_params, op_params) -> List[Dict[str, Any]]:
    def deref(p: Dict[str, Any]) -> Dict[str, Any]:
        if "$ref" in p:
            target = resolve_ref(sw, p["$ref"])
            return dict(target or {})
        return p
    combined = {}
    for p in (path_params or []):
        p = deref(p) or {}
        key = (p.get("name"), p.get("in"))
        combined[key] = p
    for p in (op_params or []):
        p = deref(p) or {}
        key = (p.get("name"), p.get("in"))
        combined[key] = p
    return list(combined.values())

def format_parameters_oas2(params: List[Dict[str, Any]]) -> str:
    out = []
    for p in params or []:
        name = p.get("name")
        loc = p.get("in")
        req = p.get("required")
        typ = p.get("type") or ((p.get("schema") or {}).get("type"))
        if not typ and "schema" in p and "$ref" in p.get("schema", {}):
            typ = p["schema"]["$ref"].split("/")[-1]
        desc = clean(p.get("description"))
        piece = f"{name} ({loc}, {'required' if req else 'optional'})"
        if typ:
            piece += f": {typ}"
        if desc:
            piece += f" - {desc}"
        out.append(piece)
    return flat_join(out)

def collect_responses_common(doc: Dict[str, Any], op: Dict[str, Any]) -> str:
    res = op.get("responses") or {}
    items = []
    for code, r in res.items():
        if isinstance(r, dict) and "$ref" in r:
            r = resolve_ref(doc, r["$ref"]) or {}
        desc = clean((r or {}).get("description"))
        items.append(f"{code}: {desc}" if desc else f"{code}")
    return flat_join(items)

# ---------------- OAS3 extraction ----------------

def get_info_oas3(oa: Dict[str, Any]) -> Dict[str, str]:
    info = oa.get("info", {}) or {}
    host = basePath = schemes = ""
    servers = oa.get("servers") or []
    if servers:
        url = clean((servers[0] or {}).get("url"))
        if url:
            try:
                u = urlparse(url)
                schemes = u.scheme
                host = u.netloc
                basePath = u.path
            except Exception:
                pass
    return {
        "api_title": clean(info.get("title")),
        "api_version": clean(info.get("version")),
        "api_description": clean(info.get("description")),
        "api_termsOfService": clean(info.get("termsOfService")),
        "api_contact": clean((info.get("contact") or {}).get("email") or (info.get("contact") or {}).get("name")),
        "api_license": clean((info.get("license") or {}).get("name")),
        "host": host,
        "basePath": basePath,
        "schemes": schemes,
        "consumes": "",
        "produces": "",
        "externalDocs": clean(((oa.get("externalDocs") or {}).get("url"))),
        "spec_version": "OAS3",
    }

def merge_parameters_oas3(doc: Dict[str, Any], path_item: Dict[str, Any], op: Dict[str, Any]) -> List[Dict[str, Any]]:
    def deref(p: Dict[str, Any]) -> Dict[str, Any]:
        if "$ref" in p:
            target = resolve_ref(doc, p["$ref"])
            return dict(target or {})
        return p
    combined = {}
    for p in (path_item.get("parameters") or []):
        p = deref(p) or {}
        key = (p.get("name"), p.get("in"))
        combined[key] = p
    for p in (op.get("parameters") or []):
        p = deref(p) or {}
        key = (p.get("name"), p.get("in"))
        combined[key] = p
    return list(combined.values())

def format_parameters_oas3(params: List[Dict[str, Any]]) -> str:
    out = []
    for p in params or []:
        name = p.get("name")
        loc = p.get("in")
        req = p.get("required")
        schema = p.get("schema") or {}
        typ = schema.get("type")
        if not typ and "$ref" in schema:
            typ = schema["$ref"].split("/")[-1]
        desc = clean(p.get("description"))
        piece = f"{name} ({loc}, {'required' if req else 'optional'})"
        if typ:
            piece += f": {typ}"
        if desc:
            piece += f" - {desc}"
        out.append(piece)
    return flat_join(out)

def request_body_summary(op: Dict[str, Any]) -> Tuple[str, List[str]]:
    rb = op.get("requestBody") or {}
    if "$ref" in rb:
        return ("requestBody: see schema", [])
    req = rb.get("required", False)
    content = rb.get("content") or {}
    ctypes = list(content.keys())
    piece = ""
    if ctypes:
        piece = f"requestBody ({'required' if req else 'optional'}): " + ", ".join(ctypes)
    elif rb:
        piece = f"requestBody ({'required' if req else 'optional'})"
    return (piece, ctypes)

def response_content_types(op: Dict[str, Any]) -> List[str]:
    res = op.get("responses") or {}
    ctypes = set()
    for r in res.values():
        if isinstance(r, dict):
            content = r.get("content") or {}
            for ct in content.keys():
                ctypes.add(ct)
    return sorted(ctypes)

# ---------------- Unified extraction ----------------

CSV_FIELDS = [
    "row_type",            # api or operation
    "spec_version",        # OAS2 or OAS3
    "api_name",
    "source_file",         # file name only
    "api_title",
    "api_version",
    "host",
    "basePath",
    "schemes",
    "consumes",
    "produces",
    "api_termsOfService",
    "api_contact",
    "api_license",
    "path",
    "method",
    "operationId",
    "summary",
    "description",
    "tags",
    "tag_descriptions",
    "parameters",
    "responses",
    "security",
    "deprecated",
    "externalDocs",
]

def extract_rows(doc: Dict[str, Any], source_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    tag_desc = tag_desc_map(doc)
    file_name = Path(source_path).name  # only the file name

    if is_oas2(doc):
        info = get_info_oas2(doc)
        api_name = info.get("api_title") or file_name

        api_row = {
            "row_type": "api",
            "spec_version": info["spec_version"],
            "api_name": api_name,
            "source_file": file_name,
            "path": "",
            "method": "",
            "operationId": "",
            "summary": "",
            "description": info.get("api_description", ""),
            "tags": flat_join([t.get("name") for t in doc.get("tags", []) if t.get("name")]),
            "tag_descriptions": flat_join([f"{k}: {v}" for k, v in tag_desc.items() if v]),
            "consumes": info.get("consumes", ""),
            "produces": info.get("produces", ""),
            "parameters": "",
            "responses": "",
            "security": security_to_str(doc.get("security")),
            "deprecated": "",
            "externalDocs": info.get("externalDocs", ""),
            "api_title": info.get("api_title", ""),
            "api_version": info.get("api_version", ""),
            "host": info.get("host", ""),
            "basePath": info.get("basePath", ""),
            "schemes": info.get("schemes", ""),
            "api_termsOfService": info.get("api_termsOfService", ""),
            "api_contact": info.get("api_contact", ""),
            "api_license": info.get("api_license", ""),
        }
        rows.append(api_row)

        paths = doc.get("paths") or {}
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            path_level_params = path_item.get("parameters", [])
            for method in ["get", "put", "post", "delete", "options", "head", "patch"]:
                if method not in path_item:
                    continue
                op = path_item[method] or {}
                params = collect_parameters_oas2(doc, path_level_params, op.get("parameters", []))
                consumes = flat_join(op.get("consumes", doc.get("consumes")))
                produces = flat_join(op.get("produces", doc.get("produces")))
                tags = op.get("tags", [])
                row = {
                    "row_type": "operation",
                    "spec_version": info["spec_version"],
                    "api_name": api_name,
                    "source_file": file_name,
                    "path": path,
                    "method": method.upper(),
                    "operationId": clean(op.get("operationId")),
                    "summary": clean(op.get("summary")),
                    "description": clean(op.get("description")),
                    "tags": flat_join(tags),
                    "tag_descriptions": flat_join([f"{t}: {tag_desc.get(t, '')}" for t in tags if tag_desc.get(t)]),
                    "consumes": consumes,
                    "produces": produces,
                    "parameters": format_parameters_oas2(params),
                    "responses": collect_responses_common(doc, op),
                    "security": security_to_str(op.get("security")),
                    "deprecated": str(bool(op.get("deprecated", False))),
                    "externalDocs": clean(((op.get("externalDocs") or {}).get("url"))),
                    "api_title": info.get("api_title", ""),
                    "api_version": info.get("api_version", ""),
                    "host": info.get("host", ""),
                    "basePath": info.get("basePath", ""),
                    "schemes": info.get("schemes", ""),
                    "api_termsOfService": info.get("api_termsOfService", ""),
                    "api_contact": info.get("api_contact", ""),
                    "api_license": info.get("api_license", ""),
                }
                rows.append(row)
        return rows

    if is_oas3(doc):
        info = get_info_oas3(doc)
        api_name = info.get("api_title") or file_name

        api_consumes: set = set()
        api_produces: set = set()

        paths = doc.get("paths") or {}
        op_rows: List[Dict[str, str]] = []
        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in ["get", "put", "post", "delete", "options", "head", "patch", "trace"]:
                if method not in path_item:
                    continue
                op = path_item[method] or {}
                params = merge_parameters_oas3(doc, path_item, op)
                param_str = format_parameters_oas3(params)

                rb_str, rb_ctypes = request_body_summary(op)
                if rb_ctypes:
                    api_consumes.update(rb_ctypes)
                if rb_str:
                    param_str = flat_join([param_str, rb_str]) if param_str else rb_str

                resp_ctypes = response_content_types(op)
                if resp_ctypes:
                    api_produces.update(resp_ctypes)

                tags = op.get("tags", [])
                op_rows.append({
                    "row_type": "operation",
                    "spec_version": info["spec_version"],
                    "api_name": api_name,
                    "source_file": file_name,
                    "path": path,
                    "method": method.upper(),
                    "operationId": clean(op.get("operationId")),
                    "summary": clean(op.get("summary")),
                    "description": clean(op.get("description")),
                    "tags": flat_join(tags),
                    "tag_descriptions": flat_join([f"{t}: {tag_desc.get(t, '')}" for t in tags if tag_desc.get(t)]),
                    "consumes": flat_join(rb_ctypes),
                    "produces": flat_join(resp_ctypes),
                    "parameters": param_str,
                    "responses": collect_responses_common(doc, op),
                    "security": security_to_str(op.get("security")),
                    "deprecated": str(bool(op.get("deprecated", False))),
                    "externalDocs": clean(((op.get("externalDocs") or {}).get("url"))),
                    "api_title": info.get("api_title", ""),
                    "api_version": info.get("api_version", ""),
                    "host": info.get("host", ""),
                    "basePath": info.get("basePath", ""),
                    "schemes": info.get("schemes", ""),
                    "api_termsOfService": info.get("api_termsOfService", ""),
                    "api_contact": info.get("api_contact", ""),
                    "api_license": info.get("api_license", ""),
                })

        api_row = {
            "row_type": "api",
            "spec_version": info["spec_version"],
            "api_name": api_name,
            "source_file": file_name,
            "path": "",
            "method": "",
            "operationId": "",
            "summary": "",
            "description": info.get("api_description", ""),
            "tags": flat_join([t.get("name") for t in doc.get("tags", []) if t.get("name")]),
            "tag_descriptions": flat_join([f"{k}: {v}" for k, v in tag_desc.items() if v]),
            "consumes": flat_join(sorted(api_consumes)),
            "produces": flat_join(sorted(api_produces)),
            "parameters": "",
            "responses": "",
            "security": security_to_str(doc.get("security")),
            "deprecated": "",
            "externalDocs": info.get("externalDocs", ""),
            "api_title": info.get("api_title", ""),
            "api_version": info.get("api_version", ""),
            "host": info.get("host", ""),
            "basePath": info.get("basePath", ""),
            "schemes": info.get("schemes", ""),
            "api_termsOfService": info.get("api_termsOfService", ""),
            "api_contact": info.get("api_contact", ""),
            "api_license": info.get("api_license", ""),
        }
        return [api_row] + op_rows

    return []

# ---------------- Runner ----------------

def scan_folder(folder: Path) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    for p in folder.rglob("*.json"):
        doc = read_json(p)
        if not doc:
            continue
        rows = extract_rows(doc, str(p))
        if rows:
            all_rows.extend(rows)
    return all_rows

def write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})

def main():
    ap = argparse.ArgumentParser(description="Scan folder and sub-folders for OAS 2.0 or 3.0 JSON specs and export API metadata to CSV.")
    ap.add_argument("folder", help="Folder to scan")
    ap.add_argument("-o", "--output", default="api_metadata.csv", help="Output CSV path")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    rows = scan_folder(folder)
    if not rows:
        print("No valid OAS 2.0 or 3.0 JSON specs found.", file=sys.stderr)
    write_csv(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
