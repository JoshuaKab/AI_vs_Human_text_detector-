# AI vs Human Text Detector

A FastAPI website and JSON API for the DistilBERT text classifier trained in
`Colab Notebooks/AI_text_classication.ipynb`.

## Install the trained model

The notebook saves the fine-tuned model and tokenizer to `ai_human_detector/`.
That directory is not included in this project, so the classifier cannot make
predictions until the exported model is available. Run the notebook's training
and save steps, then copy/download the complete `ai_human_detector` directory
into the project root. It should include the model weights, `config.json`, and
tokenizer files.

The model must define `id2label` with `0: HUMAN` and `1: AI` (or the same labels
at the corresponding class indices), as in the training notebook.

## Run locally

Python 3.11 is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:MODEL_PATH = ".\ai_human_detector"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>. If using a different model directory, set
`MODEL_PATH` to its path before starting the server.

If the model directory is missing, the website still loads and reports the
missing model; `POST /api/detect` returns HTTP 503 rather than producing a
placeholder prediction.

## Docker deployment

Place the exported model at `./ai_human_detector`, then build and start the
service:

```powershell
docker compose up --build -d
```

Visit <http://localhost:8000>. The model is mounted read-only into the container.
To view service logs or stop it:

```powershell
docker compose logs -f web
docker compose down
```

## Render deployment

The trained weights are large and should not be committed to Git. Upload the
contents of `ai_human_detector/` to a Hugging Face model repository instead.
Then push this app's source to GitHub and create a Render Web Service from that
repository, or use the included `render.yaml` as a Blueprint.

Set these Render environment variables:

- `MODEL_REPO_ID`: the Hugging Face model repository, for example
  `your-account/ai-human-detector`.
- `HF_TOKEN`: only required for a private Hugging Face model repository; use a
  read-only token and add it as a secret in Render.

The Blueprint uses `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Choose a
Render instance with at least 2 GB RAM for loading the model. Once deployment
finishes, open the service URL and confirm `/api/health` reports
`"model_ready": true`.

The API accepts `POST /api/detect` with JSON such as
`{"text":"A passage to analyze."}`. Text must contain at least one non-space
character and may be up to 10,000 characters. Responses include the predicted
label, model confidence, and class scores. The model tokenizes at 256 tokens,
matching the notebook's training setup.

## Limitations

This is an experimental classifier. Model confidence is not a calibrated
probability or proof of authorship, and performance may differ substantially
outside the training data. Do not use it as the sole basis for consequential
decisions.
