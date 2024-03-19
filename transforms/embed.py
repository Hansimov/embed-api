import os

from typing import Union

from tclogger import logger
from transformers import AutoModel
from numpy.linalg import norm

from configs.envs import ENVS

os.environ["HF_ENDPOINT"] = ENVS["HF_ENDPOINT"]
os.environ["HF_TOKEN"] = ENVS["HF_TOKEN"]


def cosine_similarity(a, b):
    return (a @ b.T) / (norm(a) * norm(b))


class JinaAIEmbedder:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v2-base-zh"):
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

    def encode(self, text: Union[str, list[str]]):
        if isinstance(text, str):
            text = [text]
        return self.model.encode(text)


if __name__ == "__main__":
    embedder = JinaAIEmbedder()
    text = ["How is the weather today?", "今天天气怎么样?"]
    # text = "How is the weather today?"
    embeddings = embedder.encode(text)
    logger.success(embeddings)
    # print(cosine_similarity(embeddings[0], embeddings[1]))
