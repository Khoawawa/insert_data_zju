"""
Async Deobfuscation Pipeline (LangChain + LightRAG)
---------------------------------------------------
Components:
  1) Detect obfuscation type + concise guidance (async)
  2) RAG exemplars via your LightRAG instance loaded by load_existing_rag (async)
  3) Deobfuscate using (code, type, guidance, exemplars) (async)
  4) Evaluate cyclomatic complexity; if no drop, refine and retry (≤3) (async)

Notes:
- Uses concise summaries, not full chain-of-thought.
- Plug your existing 'load_existing_rag' exactly as you provided.
"""

import os
import re
import json
import asyncio
from typing import Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

# LangChain async LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# LightRAG
from lightrag import LightRAG, QueryParam
# from lightrag_xsz import load_existing_rag
from dotenv import load_dotenv
from langchain_core.runnables import Runnable

load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
LOCAL_URL = os.getenv("LOCAL_URL")
RERANK_URL = os.getenv("RERANK_URL")
EMBEDDER_URL = os.getenv("EMBEDDER_URL")
WORKING_DIR1 = "./lightrag_xsz"
WORKING_DIR = "./lightrag_db_js_obf_bge_m3"

from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status


# -----------------------------
# Utility: Cyclomatic Complexity
# -----------------------------

# --- add once near your imports ---
from typing import Type, TypeVar, Union

T = TypeVar("T")

def cast_or_parse(obj: Union[dict, T], model: Type[T]) -> T:
    return obj if isinstance(obj, model) else model(**obj)

import asyncio, json, shutil, tempfile, os
from typing import Optional

NODE_BIN = shutil.which("node") or "node"  # adjust if needed
COMPLEXITY_JS = os.path.abspath("complexity.js")  # path to the file above


async def cyclomatic_complexity_node(js_code: str) -> Optional[int]:
    """
    Run Node/Esprima-backed cyclomatic complexity.
    Returns an int, or None on failure.
    """
    if not os.path.exists(COMPLEXITY_JS):
        return None  # script missing

    try:
        proc = await asyncio.create_subprocess_exec(
            NODE_BIN, COMPLEXITY_JS,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(js_code.encode("utf-8"))
        if not out:
            return None
        # Use the last non-empty line (some environments prepend logs)
        lines = [ln for ln in out.decode("utf-8", errors="ignore").splitlines() if ln.strip()]
        data = json.loads(lines[-1])
        return data.get("cyclomatic")
    except Exception:
        return None


_STRING_RE = re.compile(
    r"""('([^'\\]|\\.)*'|"([^"\\]|\\.)*"|`([^`\\]|\\.)*`)""",
    flags=re.DOTALL,
)


def _strip_strings_and_comments(code: str) -> str:
    code_wo_comments = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code_wo_comments = re.sub(r"//.*?$", "", code_wo_comments, flags=re.MULTILINE)
    return _STRING_RE.sub("", code_wo_comments)


def cyclomatic_complexity(js_code: str) -> int:
    """
    Approximate metric, consistent for comparison:
      base = 1
      +1 for: if, for, while, case, catch, ||, &&, ?:, ??, do
    """
    code = _strip_strings_and_comments(js_code)
    base = 1
    keyword_counts = 0
    keyword_counts += len(re.findall(r"\bif\b", code))
    keyword_counts += len(re.findall(r"\bfor\b", code))
    keyword_counts += len(re.findall(r"\bwhile\b", code))
    keyword_counts += len(re.findall(r"\bcase\b", code))
    keyword_counts += len(re.findall(r"\bcatch\b", code))
    keyword_counts += len(re.findall(r"\bdo\b\s*\{?", code))
    keyword_counts += len(re.findall(r"\?\s*[^:]+:", code))  # ternary
    keyword_counts += len(re.findall(r"\|\|", code))
    keyword_counts += len(re.findall(r"&&", code))
    keyword_counts += len(re.findall(r"\?\?", code))
    return base + keyword_counts


# -----------------------------
# Schemas for Structured Output
# -----------------------------

class DetectOutput(BaseModel):
    obfuscation_type: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    guidance_checklist: List[str]
    brief_rationale: str


class DeobfOutput(BaseModel):
    deobfuscated_code: str
    reasoning_summary: List[str]


# -----------------------------
# LLM setup (async)
# -----------------------------

def make_llm(model_name: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct", temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=model_name,
        base_url=OPENAI_BASE_URL,
        temperature=0.7,
    )


# -----------------------------------------
# Component 1: Detect obfuscation + guidance
# -----------------------------------------

def build_detect_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=DetectOutput)
    system = (
        "You are a senior JavaScript reverse engineer. "
        "Given obfuscated JS, identify likely obfuscation types and produce a concise checklist to guide deobfuscation. "
        "Give detailed Chain-of-thought instructions, based on the combinations of obfuscation types and combine the instructions"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user",
             """Obfuscated JavaScript:{code}
Common obfuscation families:
- Identifier renaming/minification
- String literal encoding/packing (Base64, hex, array-of-chars, RC4-like unpackers)
- Control-flow flattening (dispatcher, opaque predicates)
- Dead code insertion / bogus control flow
- Function inlining/outlining, proxy functions
- Object mapping / property indirection
- Self-defending / anti-tamper / eval hooks
- Domain-specific packers (JJEncode, AAEncode, Packer, Obfuscator.io patterns)

Return STRICT JSON:
{format_instructions}
"""
             ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parse

# --------------------------------------
# Component 2: RAG code-pair retrieval
# --------------------------------------

RAG_EXEMPLAR_PROMPT = """You will receive retrieved context from a code knowledge base.
Task: Extract up to {k} EXEMPLAR code pairs that BEST MATCH the target’s obfuscation patterns.

Inputs:
- Declared obfuscation types: {obf_type}
- Target obfuscated code:{target_code}
MATCHING CRITERIA
[Weight 4] TYPE ALIGNMENT with {obf_type}.
[Weight 3] STRUCTURAL SIGNALS overlap:
  - string table + indexer (e.g., _arr[idx], rotator/shift loop)
  - control-flow flattening (switch(dispatch), state variable, jump table)
  - property indirection (map object → method names)
  - encoded strings (hex \\x.., \\u...., Base64, custom decoder)
  - self-defending / anti-tamper wrappers, eval hooks
  - dead code / opaque predicates
[Weight 2] LEXICAL CUES: _0x[0-9a-f]+, one-letter params, etc.
[Penalty -3] Irrelevant APIs/contexts not present in the target.

OUTPUT (STRICT JSON ONLY):
{{
  "exemplars": [
    {{
      "id": "<doc id or path if available>",
      "obfuscated": "<obfuscated snippet>",
      "clean": "<corresponding deobfuscated snippet>",
      "short_match_notes": ["array-rotator","_0x.. indexer","hex literals"],
      "align_features": {{
        "target_tokens": ["<t1>","<t2>"],
        "candidate_tokens": ["<c1>","<c2>"]
      }},
      "match_score": 0,
      "confidence": 0.0,
      "source": "<optional>"
    }}
    // up to {k} items, highest score first
  ]
}}

RULES
- Use ONLY content present in retrieved context; do NOT invent code.
- Keep notes compact. No prose outside JSON.
"""


async def rag_fetch_code_pairs_async(
        rag: LightRAG,
        obf_type: List[str],
        target_code: str,
        k: int = 4,
        mode: str = "hybrid",
        search_k: int = 12,
) -> List[Dict[str, str]]:
    """
    Ask LightRAG to retrieve + let the LLM emit JSON exemplars extracted from the retrieved context.
    We then parse and normalize to the {obfuscated, clean, note} schema used downstream.
    """
    query = RAG_EXEMPLAR_PROMPT.format(
        obf_type=", ".join(obf_type),
        target_code=target_code,
        k=k
    )
    # QueryParam knobs: adjust to your LightRAG config
    param = QueryParam(mode=mode, top_k=search_k)
    resp = await rag.aquery(query, param=param)

    # resp may be a string or an object; normalize to string
    text = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)

    # Extract first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        print("[RAG] No JSON found in response.")
        return []

    try:
        data = json.loads(match.group(0))
    except Exception as e:
        print(f"[RAG] JSON parse error: {e}")
        return []

    exemplars = []
    for item in data.get("exemplars", [])[:k]:
        obf = item.get("obfuscated", "") or ""
        clean = item.get("clean", "") or ""
        notes = item.get("short_match_notes", [])
        note = ", ".join(notes) if isinstance(notes, list) else str(notes)
        if obf.strip() and clean.strip():
            exemplars.append({"obfuscated": obf, "clean": clean, "note": note})
    return exemplars


# ------------------------------------------
# Component 3: Deobfuscate with structured IO
# ------------------------------------------

def build_deobfuscate_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=DeobfOutput)
    system = (
        "You are a JavaScript deobfuscation assistant. "
        "Transform the input code into readable, behavior-preserving JS. "
        "Provide only concise transformation summaries (no internal chain-of-thought)."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user",
             """Task: Deobfuscate the code while preserving semantics.

Inputs:
- Obfuscation type(s): {obf_type}
- Deobfuscation checklist (concise):
{guidance_bullets}

- Exemplars (obfuscated → clean):
{exemplars}

- Target obfuscated code:{code}
Transformation rules (follow strictly):
1) Preserve behavior; simplify only when semantics are provably unchanged.
2) Replace string decoders/indirection with direct literals where determinable.
3) Remove dead code and opaque predicates that are constants.
4) Inline one-layer proxy wrappers when trivial.
5) Undo mechanical control-flow flattening where feasible.
6) Rename identifiers descriptively.
7) No libraries or I/O additions.

Return STRICT JSON:
{format_instructions}
"""
             ),
        ]
    ).partial(format_instructions=JsonOutputParser(pydantic_object=DeobfOutput).get_format_instructions())
    return prompt | llm | parser

# ----------------------------------------------------
# Component 4: Evaluation + refinement (loop controller)
# ----------------------------------------------------



REFINER_SYSTEM = (
    "You are a concise JS refiner. Given evaluation feedback, produce a short additive hint to refine the checklist. "
    "Do not provide chain-of-thought—just a targeted guidance sentence."
)
REFINER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REFINER_SYSTEM),
        ("user",
         """Context:
- Obfuscation type(s): {obf_type}
- Current checklist:
{guidance_bullets}

- Evaluation result:
{evaluation_msg}

Return a single short line beginning with 'Refinement: ' that suggests one extra actionable step."""
         ),
    ]
)


def build_refiner_chain(llm: ChatOpenAI) -> Runnable:
    # Return the LLM content text
    return REFINER_PROMPT | llm | (lambda x: x.content.strip())


# -----------------------------
# Orchestration (async)
# -----------------------------

def format_bullets(items: List[str]) -> str:
    return "\n".join(f"- {it}" for it in items)


def format_exemplars(pairs: List[Dict[str, str]]) -> str:
    if not pairs:
        return "(none supplied)"
    blocks = []
    for i, p in enumerate(pairs, 1):
        note = f"\nNote: {p.get('note', '')}" if p.get("note") else ""
        blocks.append(
            f"Example {i} (obfuscated):\n```\n{p['obfuscated']}\n```\n"
            f"→ (clean):\n```\n{p['clean']}\n```\n{note}"
        )
    return "\n\n".join(blocks)


async def run_pipeline_async(
        obfuscated_js: str,
        working_dir: str,
        model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        k_exemplars: int = 4,
        max_iters: int = 3,
) -> Dict[str, object]:
    """
    Returns:
      {
        'obfuscation_type': [...],
        'guidance': [...],
        'attempts': [...],
        'final': { 'status': 'success'|'stopped_no_improvement', 'best_code': str, 'best_summary': [...] }
      }
    """
    # Init RAG
    rag = await load_existing_rag(working_dir)

    # LLMs
    llm = make_llm(model_name=model, temperature=0.2)
    detect_chain = build_detect_chain(llm)
    deobf_chain = build_deobfuscate_chain(llm)
    refiner_chain = build_refiner_chain(llm)

    # Component 1
    detect_raw = await detect_chain.ainvoke({"code": obfuscated_js})
    detect = cast_or_parse(detect_raw, DetectOutput)
    obf_type = detect.obfuscation_type
    guidance = list(detect.guidance_checklist)

    # Component 2 (RAG)
    exemplars = await rag_fetch_code_pairs_async(
        rag=rag,
        obf_type=obf_type,
        target_code=obfuscated_js,
        k=k_exemplars,
        mode="hybrid",
        search_k=max(8, k_exemplars * 3),
    )

    # Iterate 3 & 4
    attempts = []
    best = {"score_delta": float("-inf"), "code": "", "summary": []}
    # base_cplx = cyclomatic_complexity(obfuscated_js)
    base_cplx = await cyclomatic_complexity_node(obfuscated_js)
    if base_cplx is None:
        # fallback to your previous Python approximation
        base_cplx = cyclomatic_complexity(obfuscated_js)
    for i in range(1, max_iters + 1):
        # deobf = await deobf_chain.ainvoke({
        #     "obf_type": ", ".join(obf_type),
        #     "guidance_bullets": format_bullets(guidance),
        #     "exemplars": format_exemplars(exemplars),
        #     "code": obfuscated_js
        # })

        # deobf_code = deobf.deobfuscated_code
        deobf_raw = await deobf_chain.ainvoke({
            "obf_type": ", ".join(obf_type),
            "guidance_bullets": format_bullets(guidance),
            "exemplars": format_exemplars(exemplars),
            "code": obfuscated_js
        })
        deobf = cast_or_parse(deobf_raw, DeobfOutput)
        deobf_code = deobf.deobfuscated_code

        # deobf_cplx = cyclomatic_complexity(deobf_code)
        deobf_cplx = await cyclomatic_complexity_node(deobf_code)
        if deobf_cplx is None:
            deobf_cplx = cyclomatic_complexity(deobf_code)
        improved = deobf_cplx < base_cplx
        eval_msg = (
            f"Cyclomatic complexity: obfuscated={base_cplx}, deobfuscated={deobf_cplx}. "
            f"{'Improved ✅' if improved else 'No improvement ❌'}"
        )

        attempt_info = {
            "iteration": i,
            "deobfuscated_code": deobf_code,
            "reasoning_summary": deobf.reasoning_summary,
            "complexity_obf": base_cplx,
            "complexity_deobf": deobf_cplx,
            "improved": improved,
            "refinement_msg": None,
        }
        attempts.append(attempt_info)

        delta = base_cplx - deobf_cplx
        if delta > best["score_delta"]:
            best.update({"score_delta": delta, "code": deobf_code, "summary": deobf.reasoning_summary})

        if improved:
            break

        if i < max_iters:
            refinement = await refiner_chain.ainvoke({
                "obf_type": ", ".join(obf_type),
                "guidance_bullets": format_bullets(guidance),
                "evaluation_msg": eval_msg
            })
            attempts[-1]["refinement_msg"] = refinement
            refinement_line = refinement.replace("Refinement:", "").strip()
            if refinement_line:
                guidance.append(refinement_line)

    final_status = "success" if attempts and attempts[-1]["improved"] else "stopped_no_improvement"
    return {
        "obfuscation_type": obf_type,
        "guidance": guidance,
        "attempts": attempts,
        "final": {
            "status": final_status,
            "best_code": best["code"],
            "best_summary": best["summary"],
        },
    }


# -----------------------------
# Example CLI
# -----------------------------
from pathlib import Path
if __name__ == "__main__":
    file_path = Path('js/javascript/obfuscated/xiaoyu2er_leetcode-js')
    OBFUSCATED = file_path.resolve()


    # WORKING_DIR = "./rag_workdir"  # <- point to your existing LightRAG DB

    async def main():
        result = await run_pipeline_async(
            obfuscated_js=OBFUSCATED,
            working_dir=WORKING_DIR,
            model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            k_exemplars=4,
            max_iters=3,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


    asyncio.run(main())



