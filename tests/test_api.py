from pathlib import Path

import pytest
import transformers
from fastapi.testclient import TestClient

from app.main import TextDetector, app, detector


@pytest.fixture(autouse=True)
def avoid_loading_model(monkeypatch) -> None:
    monkeypatch.setattr(detector, "load", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(detector, "ready", False)


def test_homepage_is_served() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Who wrote this?" in response.text


def test_health_reports_missing_model() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_ready"] is False


def test_detection_requires_model() -> None:
    with TestClient(app) as client:
        response = client.post("/api/detect", json={"text": "A test passage"})

    assert response.status_code == 503


def test_detection_returns_prediction_when_model_is_ready(monkeypatch) -> None:
    with TestClient(app) as client:
        monkeypatch.setattr(detector, "ready", True)
        monkeypatch.setattr(
            detector,
            "predict",
            lambda _: {
                "label": "AI",
                "confidence": 0.9,
                "ai_score": 0.9,
                "human_score": 0.1,
            },
        )
        response = client.post("/api/detect", json={"text": "A test passage"})

    assert response.status_code == 200
    assert response.json()["label"] == "AI"
    assert response.json()["confidence"] == 0.9


def test_detection_rejects_blank_and_oversized_text() -> None:
    with TestClient(app) as client:
        blank = client.post("/api/detect", json={"text": "   "})
        oversized = client.post("/api/detect", json={"text": "a" * 10_001})

    assert blank.status_code == 422
    assert oversized.status_code == 422


def test_model_can_load_from_hugging_face_repository(monkeypatch) -> None:
    calls = []

    class FakeModel:
        config = type("Config", (), {"id2label": {0: "HUMAN", 1: "AI"}})()

        def to(self, _: str) -> "FakeModel":
            return self

        def eval(self) -> None:
            pass

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda source, **options: calls.append(("tokenizer", source, options)),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda source, **options: (
            calls.append(("model", source, options)) or FakeModel()
        ),
    )

    model_detector = TextDetector()
    model_detector.load(
        model_path=Path("missing-local-model"),
        model_repo_id="example/ai-human-detector",
        hf_token="read-token",
    )

    assert model_detector.ready is True
    assert calls == [
        ("tokenizer", "example/ai-human-detector", {"token": "read-token"}),
        ("model", "example/ai-human-detector", {"token": "read-token"}),
    ]
