import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)
MAX_TEXT_LENGTH = 10_000
MAX_TOKEN_LENGTH = 256
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "ai_human_detector"


class DetectionRequest(BaseModel):
    text: str = Field(max_length=MAX_TEXT_LENGTH)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank.")
        return value


class DetectionResponse(BaseModel):
    label: Literal["AI", "HUMAN"]
    confidence: float = Field(ge=0, le=1)
    ai_score: float = Field(ge=0, le=1)
    human_score: float = Field(ge=0, le=1)


class TextDetector:
    def __init__(self) -> None:
        self.ready = False
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.torch = None
        self.ai_index = 1
        self.human_index = 0

    def load(
        self,
        model_path: Path,
        model_repo_id: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        if model_repo_id is None and not model_path.is_dir():
            logger.warning("Trained model directory not found: %s", model_path)
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_source = model_repo_id or str(model_path)
        load_options = {"token": hf_token} if hf_token else {}
        tokenizer = AutoTokenizer.from_pretrained(model_source, **load_options)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source, **load_options
        )
        id2label = model.config.id2label
        labels = {
            str(label).strip().upper(): int(index)
            for index, label in id2label.items()
        }
        ai_indices = [index for label, index in labels.items() if label == "AI"]
        human_indices = [index for label, index in labels.items() if label == "HUMAN"]
        if len(ai_indices) != 1 or len(human_indices) != 1:
            raise ValueError(
                "The model must define id2label entries for both AI and HUMAN."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        model.to(self.device)
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.ai_index = ai_indices[0]
        self.human_index = human_indices[0]
        self.ready = True
        logger.info("Loaded text classifier from %s on %s", model_path, self.device)

    def predict(self, text: str) -> DetectionResponse:
        if (
            not self.ready
            or self.model is None
            or self.tokenizer is None
            or self.torch is None
        ):
            raise RuntimeError("The trained model is not loaded.")

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            scores = self.torch.softmax(logits, dim=-1)[0].cpu().tolist()

        ai_score = float(scores[self.ai_index])
        human_score = float(scores[self.human_index])
        label = "AI" if ai_score >= human_score else "HUMAN"
        return DetectionResponse(
            label=label,
            confidence=max(ai_score, human_score),
            ai_score=ai_score,
            human_score=human_score,
        )


detector = TextDetector()


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_path = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    detector.load(
        model_path,
        model_repo_id=os.environ.get("MODEL_REPO_ID"),
        hf_token=os.environ.get("HF_TOKEN"),
    )
    yield


app = FastAPI(
    title="AI vs Human Text Detector",
    description="Classifies text with the project's fine-tuned DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ready" if detector.ready else "model_missing",
        "model_ready": detector.ready,
        "message": (
            "The text detector is ready."
            if detector.ready
            else "Install the trained model in ai_human_detector or set MODEL_PATH."
        ),
    }


@app.post("/api/detect", response_model=DetectionResponse)
def detect(request: DetectionRequest) -> DetectionResponse:
    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail="The trained model is not installed. See the deployment instructions.",
        )
    return detector.predict(request.text)
