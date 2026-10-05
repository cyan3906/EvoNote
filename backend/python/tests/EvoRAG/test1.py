from openai import OpenAI

# -------------------------- 配置你的中转站信息 --------------------------
API_KEY = "sk-nmjbdjNwStfim8V9gaLaA7CiJNjHAhYOayGlY7nUCJsuhhZQ"
BASE_URL = "https://api.quickrouter.ai/v1"  # 中转站地址，末尾必须带 /v1
MODEL_NAME = "text-embedding-3-large" # 向量模型名称，确认中转站支持这个名字
# -----------------------------------------------------------------------

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=30.0
)

def get_embedding(text: str | list[str]) -> list[float] | list[list[float]]:
    """
    获取文本向量
    :param text: 单个字符串 / 字符串列表（批量）
    :return: 向量数组，批量输入返回向量列表
    """
    resp = client.embeddings.create(
        input=text,
        model=MODEL_NAME
    )
    if isinstance(text, str):
        return resp.data[0].embedding
    else:
        return [item.embedding for item in resp.data]


if __name__ == "__main__":
    # 单个文本测试
    text = "MVCC多版本并发控制原理"
    vec = get_embedding(text)
    print(f"向量维度: {len(vec)}")
    print(f"向量前10个值: {vec[:10]}")

    # 批量文本测试
    texts = [
        "QUIC协议基于UDP，解决TCP队头阻塞",
        "RAG检索增强生成可以降低大模型幻觉"
    ]
    vecs = get_embedding(texts)
    print(f"\n批量获取，共{len(vecs)}个向量")