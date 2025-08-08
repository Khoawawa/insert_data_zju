import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from lightrag.llm.openai import openai_embed
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL")
async def main():
    vecs = await openai_embed(
                ['chunk1', 'chunk2'],
                model="BAAI/bge-m3",
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
            )
    print(vecs)
    # url = "https://api.siliconflow.cn/v1/embeddings"

    # payload = {
    #     "model": "BAAI/bge-m3",
    #     "input": "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!"
    # }
    # headers = {
    #     "Authorization": f"Bearer {OPENAI_API_KEY}",
    #     "Content-Type": "application/json"
    # }

    # response = requests.post(url, json=payload, headers=headers)

    # print(response.json())

if __name__ == "__main__":
    asyncio.run(main())
