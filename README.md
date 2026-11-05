# Skin Check AI

An AI skin-health assistant that combines **your own trained image classifier** with **Claude vision**. Upload (or take) a photo of a skin concern, add a few details, and get:

- **Trained model prediction**: an EfficientNet model you train yourself (see `GUIDE.md`) predicts the condition with a confidence score
- **AI review**: Claude checks the model's prediction against the photo and ranks possible conditions with reasons
- **What the AI sees** in the photo, common causes, a gentle care routine and lifestyle tips
- **Urgency level** (routine / see a doctor soon / get help now) with clear "when to see a doctor" advice
- **Safety rules** that raise the urgency from reported symptoms (e.g. fever + spreading redness, a spot that bleeds or changes, rashes near the eye)
- A **downloadable report** and a **follow-up chat** about the result
- **Feedback** page and an **Admin** dashboard with analytics and CSV export

> ⚠️ Educational tool, not a medical diagnosis. Always consult a doctor or dermatologist.

## Tech stack
Python · PyTorch + timm (training, Google Colab) · ONNX + onnxruntime (inference) · Streamlit · Claude vision via the Anthropic Python SDK · Pillow · pandas · Plotly · SQLite (local) or MySQL/TiDB (cloud)

**👉 Step-by-step from training to deployment: [GUIDE.md](GUIDE.md)**

## How it works
1. **Image prep** (`analyzer.prepare_image`): fixes rotation, strips EXIF metadata (location, camera), resizes to ≤1568 px, re-encodes as JPEG.
2. **Trained model** (`classifier.py`): your ONNX model predicts the top 3 conditions. It works even without an API key.
3. **AI analysis** (`analyzer.analyze_image`): the photo, the details and the model's prediction are sent to Claude with a **JSON schema** (structured outputs). Claude says whether it agrees with the model and writes the report.
4. **Safety layer** (`analyzer.red_flag_rules`): rule-based checks on the reported symptoms; the final urgency is the **higher** of the AI's and the rules'.
5. **Display + storage**: results are shown with charts. Only anonymous metadata (age band, area, result, urgency) is saved. **Photos are never stored.**
6. **Follow-up chat** (`analyzer.ask_followup`): a multi-turn conversation that keeps the photo and analysis as context.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...   # macOS/Linux: export ANTHROPIC_API_KEY=sk-ant-...
streamlit run App.py
```
Get an API key at https://console.anthropic.com. Without a key, the app still shows your trained model's prediction; the AI report and chat need the key.

## Configuration (environment variables / Streamlit Secrets)
| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | - | **Required** for AI analysis |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Model used |
| `ADMIN_USER` / `ADMIN_PASS` | `admin` / `admin@skin-check` | Admin login. **Change before deploying.** |
| `APP_ACCESS_CODE` | *(unset)* | If set, users must enter this code to run analyses |
| `MAX_ANALYSES_PER_SESSION` | `0` (no limit) | Optional per-visitor limit |
| `DAILY_ANALYSIS_LIMIT` | `0` (no limit) | Optional whole-app daily limit |
| `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB` | *(unset → SQLite)* | Cloud database (e.g. TiDB) |
| `MYSQL_SSL` / `MYSQL_SSL_CA` | *(unset)* | Encrypted DB connection (`MYSQL_SSL = "true"` for TiDB) |

### Streamlit Cloud Secrets example
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ADMIN_USER = "yourname"
ADMIN_PASS = "a-strong-password"
MYSQL_HOST = "gateway01.us-east-1.prod.aws.tidbcloud.com"
MYSQL_PORT = "4000"
MYSQL_USER = "xxxxxxxx.root"
MYSQL_PASSWORD = "your-tidb-password"
MYSQL_DB = "test"
MYSQL_SSL = "true"
```

## Project layout
| File | Role |
|---|---|
| `App.py` | Streamlit UI: Analyze, Feedback, About, Admin |
| `ui.py` | Visual design: custom CSS, hero, cards, urgency banner (light + dark mode) |
| `.streamlit/config.toml` | Theme: colours, font, rounded corners, light/dark |
| `analyzer.py` | Image prep, Claude calls, JSON schema, safety rules, follow-up chat, report |
| `classifier.py` | Runs your trained ONNX model (same preprocessing as training) |
| `db.py` | SQLite / MySQL storage (no photos) |
| `models/` | Put `skin_model.onnx` + `labels.json` here |
| `training/train.py` | Training script: transfer learning, metrics, ONNX export |
| `training/Train_Skin_Model.ipynb` | Google Colab notebook to train on a free GPU |
