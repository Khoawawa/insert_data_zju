# ingest_lightrag_bge_local.py
import os
import asyncio
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import AsyncOpenAI, openai_embed
from lightrag.utils import EmbeddingFunc

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY == "sk-ltmsqvvytksvhvefzzwtyrqlpmdyxiyxnoglxpfeggswbezm", "Please set OPENAI_API_KEY in .env file"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
assert OPENAI_BASE_URL == "https://api.siliconflow.cn/v1", "Please set OPENAI_BASE_URL in .env file"
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")
LOCAL_URL = os.getenv("LOCAL_URL")
RERANK_URL = os.getenv("RERANK_URL")
EMBEDDER_URL = os.getenv("EMBEDDER_URL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ---- High performance settings ----
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "8"))  # Increased concurrent LLM calls
LLM_SLEEP = float(os.getenv("LLM_SLEEP", "0"))  # No delay
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))  # Fewer retries

EMBED_MAX_BATCH = int(os.getenv("EMBED_MAX_BATCH", "64"))  # Larger batches
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "8"))  # Higher concurrency
EMBED_SLEEP = float(os.getenv("EMBED_SLEEP", "0"))  # No delay
EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-m3 = 1024

INSERT_BATCH = int(os.getenv("INSERT_BATCH", "64"))  # Larger batches
INSERT_SLEEP = float(os.getenv("INSERT_SLEEP", "0"))  # No delay

# ----- Your paths & types -----
ORIGINAL_ROOT = Path("./js/javascript/xiaoyu2er_leetcode-js").resolve()
OBF_ROOT_TEMPLATE = "./js/javascript/obfuscated/{obfuscation_type}/javascript/xiaoyu2er_leetcode-js"
OBFUSCATION_TYPES = ["data", "encode", "logic", "randomization", "synthesized_high", "trick"]

WORKING_DIR = "./lightrag_db_js_obf_bge_m3"
LANGUAGE = "JavaScript"


def reset_working_dir():
    """Reset working dir; do NOT pre-create vdb_*.json files."""
    if os.path.exists(WORKING_DIR):
        import shutil
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)


def stable_id(*parts: str) -> str:
    return hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()[:12]


def list_js(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.js") if p.is_file()]


def rel_to(root: Path, p: Path) -> str:
    return str(p.relative_to(root))


def build_doc(pair_id: str, orig: Path, obf: Path, types: List[str], orig_code: str, obf_code: str) -> str:
    header = {
        "PAIR_ID": pair_id,
        "LANGUAGE": LANGUAGE,
        "ORIGINAL_PATH": str(orig),
        "OBFUSCATED_PATH": str(obf),
        "OBFUSCATION_TYPES": types,
    }
    h = "\n".join(f"{k}: {json.dumps(v) if isinstance(v, (list, dict)) else v}" for k, v in header.items())
    return (
        f"{h}\n\n=== ORIGINAL ({LANGUAGE}) ===\n```js\n{orig_code}\n```\n\n"
        f"=== OBFUSCATED ({LANGUAGE}) ===\n```js\n{obf_code}\n```\n"
    )


# ---------- Throttled, backoff LLM ----------
async_openai = AsyncOpenAI(base_url=LOCAL_URL, api_key=OPENAI_API_KEY)
_llm_sema = asyncio.Semaphore(LLM_CONCURRENCY)


async def my_llm_complete(prompt: str, **kwargs) -> str:
    """Fast LLM completion with minimal retry."""
    async with _llm_sema:
        resp = await async_openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content or ""


# ---------- Throttled, backoff REMOTE EMBEDDER ----------
_embed_sema = asyncio.Semaphore(EMBED_CONCURRENCY)


async def embedding_func(texts: List[str]):
    """Fast chunked remote embeddings with minimal retry."""
    outputs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_MAX_BATCH):
        chunk = texts[i:i + EMBED_MAX_BATCH]
        async with _embed_sema:
            vecs = await openai_embed(
                chunk,
                model="BAAI/bge-m3",
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
            )
            outputs.extend(vecs)
    return outputs

async def main():
    # ⚠️ Resetting the DB forces re-extraction (more LLM calls). Keep if you really need a clean rebuild.
    print("重置工作目录...")
    reset_working_dir()

    # Check dirs
    print(f"\n检查目录结构:")
    print(f"原始代码目录: {ORIGINAL_ROOT}")
    if not ORIGINAL_ROOT.exists():
        print(f"警告: 原始代码目录不存在!")
        os.makedirs(ORIGINAL_ROOT, exist_ok=True)

    for t in OBFUSCATION_TYPES:
        obf_path = Path(OBF_ROOT_TEMPLATE.format(obfuscation_type=t))
        print(f"混淆代码目录 ({t}): {obf_path}")
        if not obf_path.exists():
            print(f"警告: 混淆代码目录不存在: {obf_path}")
            os.makedirs(obf_path, exist_ok=True)

    # LightRAG with high parallelism for speed
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            func=embedding_func,
            embedding_dim=EMBED_DIM,
            max_token_size=8192,
        ),
        llm_model_func=my_llm_complete,
        max_parallel_insert=1,  # Increased for faster processing
        # enable later; it adds more LLM work at query time
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Build docs
    print("\n查找文件:")
    orig_files = list_js(ORIGINAL_ROOT)
    print(f"在原始目录中找到 {len(orig_files)} 个 JavaScript 文件")

    texts, ids = [], []
    obf_roots: Dict[str, Path] = {t: Path(OBF_ROOT_TEMPLATE.format(obfuscation_type=t)).resolve()
                                  for t in OBFUSCATION_TYPES}

    if len(orig_files) == 0:
        print(f"错误: 在 {ORIGINAL_ROOT} 中没有找到任何 .js 文件")

    print("\n处理文件对:")
    total_files = len(orig_files)
    for file_idx, op in enumerate(orig_files, 1):
        rel = rel_to(ORIGINAL_ROOT, op)
        if not op.exists():
            print(f"警告: 原始文件不存在: {op}")
            continue
        print(f"\n[{file_idx}/{total_files}] 处理原始文件: {rel}")
        ocode = op.read_text(encoding="utf-8", errors="ignore")

        pairs_found = 0
        for t, root in obf_roots.items():
            p = root / rel
            if not p.exists():
                print(f"  - 没有找到对应的混淆文件 ({t}): {p}")
                continue
            ucode = p.read_text(encoding="utf-8", errors="ignore")
            pid = stable_id(rel, t)
            texts.append(build_doc(pid, op, p, [t], ocode, ucode))
            ids.append(pid)
            pairs_found += 1

        print(f"  完成: {pairs_found} 个混淆对已处理 ({(file_idx / total_files * 100):.1f}% 完成)")

    if not texts:
        print("\n错误: 没有找到任何文件对!")
        print(
            "请确保:\n1. 原始代码目录存在并包含 .js 文件\n2. 混淆代码目录结构正确\n3. 每个原始文件在混淆目录中都有对应的文件")
        return

    # Insert with progress tracking
    total_batches = (len(texts) + INSERT_BATCH - 1) // INSERT_BATCH
    print(f"\n开始插入文档 (总计 {len(texts)} 个文档, 分 {total_batches} 批处理)")

    start_time = time.time()
    docs_inserted = 0

    for i in range(0, len(texts), INSERT_BATCH):
        batch_num = i // INSERT_BATCH + 1
        batch_texts = texts[i:i + INSERT_BATCH]
        batch_ids = ids[i:i + INSERT_BATCH]

        batch_start = time.time()
        await rag.ainsert(batch_texts, ids=batch_ids)
        batch_time = time.time() - batch_start

        docs_inserted += len(batch_texts)
        progress = docs_inserted / len(texts) * 100
        elapsed = time.time() - start_time
        docs_per_sec = docs_inserted / elapsed if elapsed > 0 else 0

        print(f"批次 {batch_num}/{total_batches} 完成: "
              f"{docs_inserted}/{len(texts)} 文档 ({progress:.1f}%) | "
              f"速度: {docs_per_sec:.1f} 文档/秒 | "
              f"本批用时: {batch_time:.1f}秒")

    total_time = time.time() - start_time
    final_speed = len(texts) / total_time if total_time > 0 else 0

    await rag.finalize_storages()
    print(f"\n插入完成:")
    print(f"- 总文档数: {len(texts)}")
    print(f"- 总用时: {total_time:.1f} 秒")
    print(f"- 平均速度: {final_speed:.1f} 文档/秒")
    print(f"- 存储位置: {WORKING_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
