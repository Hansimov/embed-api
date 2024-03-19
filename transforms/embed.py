import os

import numpy as np
import torch

from pathlib import Path
from typing import Union

from huggingface_hub import hf_hub_download
from numpy.linalg import norm
from onnxruntime import InferenceSession
from tclogger import logger
from transformers import AutoTokenizer, AutoModel

from configs.envs import ENVS
from configs.constants import AVAILABLE_MODELS

if ENVS["HF_ENDPOINT"]:
    os.environ["HF_ENDPOINT"] = ENVS["HF_ENDPOINT"]
os.environ["HF_TOKEN"] = ENVS["HF_TOKEN"]


def cosine_similarity(a, b):
    return (a @ b.T) / (norm(a) * norm(b))


class JinaAIOnnxEmbedder:
    """https://huggingface.co/jinaai/jina-embeddings-v2-base-zh/discussions/6#65bc55a854ab5eb7b6300893"""

    def __init__(self):
        self.repo_name = "jinaai/jina-embeddings-v2-base-zh"
        self.download_model()
        self.load_model()

    def download_model(self):
        self.onnx_folder = Path(__file__).parent
        self.onnx_filename = "onnx/model_quantized.onnx"
        self.onnx_path = self.onnx_folder / self.onnx_filename
        if not self.onnx_path.exists():
            logger.note("> Downloading ONNX model")
            hf_hub_download(
                repo_id=self.repo_name,
                filename=self.onnx_filename,
                local_dir=self.onnx_folder,
                local_dir_use_symlinks=False,
            )
            logger.success(f"+ ONNX model downloaded: {self.onnx_path}")
        else:
            logger.success(f"+ ONNX model loaded: {self.onnx_path}")

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repo_name, trust_remote_code=True
        )
        self.session = InferenceSession(self.onnx_path)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors="np")
        inputs = {
            name: np.array(tensor, dtype=np.int64) for name, tensor in inputs.items()
        }
        outputs = self.session.run(
            output_names=["last_hidden_state"], input_feed=dict(inputs)
        )
        embeddings = self.mean_pooling(
            torch.from_numpy(outputs[0]), torch.from_numpy(inputs["attention_mask"])
        )
        return embeddings


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
    # embedder = JinaAIEmbedder()
    embedder = JinaAIOnnxEmbedder()
    texts = ["How is the weather today?", "今天天气怎么样?"]
    embeddings = []
    for text in texts:
        embeddings.append(embedder.encode(text))
    logger.success(embeddings)
    print(cosine_similarity(embeddings[0], embeddings[1]))

    # python -m transforms.embed
