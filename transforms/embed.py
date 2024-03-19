import os

from typing import Union

from tclogger import logger
from transformers import AutoModel
from numpy.linalg import norm

from configs.envs import ENVS
from configs.constants import AVAILABLE_MODELS

os.environ["HF_ENDPOINT"] = ENVS["HF_ENDPOINT"]
os.environ["HF_TOKEN"] = ENVS["HF_TOKEN"]


def cosine_similarity(a, b):
    return (a @ b.T) / (norm(a) * norm(b))


class JinaAIEmbedder:
    def __init__(self, model_name: str = AVAILABLE_MODELS[0]):
        self.model_name = model_name
        self.load_model()

    def check_model_name(self):
        if self.model_name not in AVAILABLE_MODELS:
            self.model_name = AVAILABLE_MODELS[0]
        return True

    def load_model(self):
        self.check_model_name()
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

    def switch_model(self, model_name: str):
        if model_name != self.model_name:
            self.model_name = model_name
            self.load_model()

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
