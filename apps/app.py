import argparse
import markdown2
import sys
import uvicorn

from pathlib import Path
from typing import Union, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from tclogger import logger, OSEnver

from transforms.embed import JinaAIEmbedder, JinaAIOnnxEmbedder
from configs.constants import AVAILABLE_MODELS

info_path = Path(__file__).parents[1] / "configs" / "info.json"
ENVER = OSEnver(info_path)


class EmbeddingApp:
    def __init__(self):
        self.app = FastAPI(
            docs_url="/",
            title=ENVER["app_name"],
            swagger_ui_parameters={"defaultModelsExpandDepth": -1},
            version=ENVER["version"],
        )
        self.embedder = JinaAIOnnxEmbedder()
        self.setup_routes()

    def get_available_models(self):
        return AVAILABLE_MODELS

    def get_readme(self):
        readme_path = Path(__file__).parents[1] / "README.md"
        with open(readme_path, "r", encoding="utf-8") as rf:
            readme_str = rf.read()
        readme_html = markdown2.markdown(
            readme_str, extras=["table", "fenced-code-blocks", "highlightjs-lang"]
        )
        return readme_html

    class CalcEmbeddingPostItem(BaseModel):
        text: Union[str, list[str]] = Field(
            default=None,
            summary="Input text(s) to embed",
        )
        model: Optional[str] = Field(
            default=AVAILABLE_MODELS[0],
            summary="Embedding model name",
        )

    def calc_embedding(self, item: CalcEmbeddingPostItem):
        logger.note(f"> Encoding text: [{item.text}]", end=" ")
        # if item.model != self.embedder.model:
        #     self.embedder.switch_model(item.model)
        embeddings = self.embedder.encode(item.text).tolist()
        logger.success(f"[{len(embeddings[0])}]")
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return embeddings

    def setup_routes(self):
        self.app.get(
            "/models",
            summary="Get available models",
        )(self.get_available_models)

        self.app.post(
            "/embedding",
            summary="Calculate embedding for input text(s)",
        )(self.calc_embedding)

        self.app.get(
            "/readme",
            summary="README of Embed API",
            response_class=HTMLResponse,
            include_in_schema=False,
        )(self.get_readme)


class ArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgParser, self).__init__(*args, **kwargs)

        self.add_argument(
            "-s",
            "--server",
            type=str,
            default=ENVER["server"],
            help=f"Server IP ({ENVER['server']}) for Embed API",
        )
        self.add_argument(
            "-p",
            "--port",
            type=int,
            default=ENVER["port"],
            help=f"Server Port ({ENVER['port']}) for Embed API",
        )

        self.args = self.parse_args(sys.argv[1:])


app = EmbeddingApp().app

if __name__ == "__main__":
    args = ArgParser().args
    uvicorn.run("__main__:app", host=args.server, port=args.port)

    # python -m apps.app
