# Skin Check AI: from zero to deployed

This guide takes you from nothing to a live app with **your own trained model** and **working AI**.

```
Photo ─► your trained model (EfficientNet, ONNX) ─► prediction + confidence      ("the eyes")
      └► Claude (vision AI) gets photo + details + model prediction
             ─► checks the prediction, writes the full report, answers questions ("the expert")
      └► safety rules on reported symptoms ─► can raise the urgency
```

Total cost: **free**, except the Claude API (a few cents per analysis, capped by a spending limit you set).

---

## Part 1. Basics (what you're building)

| Term | Meaning in this project |
|---|---|
| **Dataset** | Thousands of skin photos, sorted into one folder per condition |
| **Training** | The model looks at the photos many times (epochs) and learns which visual patterns mean which condition |
| **Transfer learning** | We start from EfficientNet, which already learned general vision from 1.2M ImageNet photos, and only teach it skin conditions. That's why a small dataset and 15 minutes are enough |
| **Train / validation / test split** | Train = learn from; validation = pick the best epoch; test = final honest score on photos the model never saw |
| **Accuracy, precision, recall, F1** | How often it's right overall; per condition: how trustworthy its "yes" is, how many real cases it finds, and a balance of both |
| **ONNX** | A portable model file. The app runs it with `onnxruntime` (≈15 MB) instead of PyTorch (≈700 MB+), so it fits on free hosting |
| **LLM / Claude** | A large AI model that can read the photo and the prediction and write a human-friendly report |

---

## Part 2. Train your model (Google Colab, free GPU)

1. Go to **https://colab.research.google.com** → **File → Upload notebook** → choose `training/Train_Skin_Model.ipynb`.
2. Menu **Runtime → Change runtime type → T4 GPU → Save**.
3. Run the cells top to bottom (**Runtime → Run all**).
   - **Step 3 (dataset)** downloads the 8-class *Skin Disease Dataset* from Kaggle. If it asks you to log in: kaggle.com → profile → **Settings → API → Create New Token**, then run `import kagglehub; kagglehub.login()` and paste the username and key.
   - To use a different dataset, upload its zip and set `DATA_ZIP`.
4. Wait about 10-20 minutes. You'll see each epoch's accuracy, then the **TEST accuracy** and a per-condition table.
5. Look at the training curve and the confusion matrix (step 5).
6. Step 6 downloads `skin_model_export.zip`. Unzip it. You need:
   - `skin_model.onnx` (≈16 MB)
   - `labels.json`
   - Also keep `metrics.json`, `confusion_matrix.png` and `training_curve.png` for your report and interviews.

**Write down your test accuracy.** That number is your model's honest score.

### Improving accuracy (optional)
- Train longer: `--epochs 30`.
- Use a bigger model: `--model convnext_tiny` (≈110 MB, too big for a normal GitHub upload; see Part 5).
- Use more or better data: more photos per condition matters most.
- Check the confusion matrix: conditions that look alike (e.g. chickenpox vs shingles) need more examples.

---

## Part 3. Put the model in the app and run it on your PC

1. Copy `skin_model.onnx` and `labels.json` into the project's **`models/`** folder.
2. In the project folder:
```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run App.py
```
3. Upload a photo. You'll see the **🧠 Trained model** prediction. It works even without an API key.

---

## Part 4. Turn on the AI (Claude)

1. Go to **https://console.anthropic.com**, sign up and add billing.
   - **Billing → set a monthly spending limit** (e.g. $5).
   - **API Keys → Create Key**, then copy it (`sk-ant-...`).
2. Test it locally:
```bash
set ANTHROPIC_API_KEY=sk-ant-your-key
streamlit run App.py
```
You now get the full report, **AI review of the model's prediction**, and follow-up chat.

**Never put the key in your code or on GitHub.**

---

## Part 5. Deploy

### 5a. GitHub
1. github.com → **New repository** (e.g. `Skin-Check-AI`).
2. **Add file → Upload files** → drag in:
   - `App.py`, `analyzer.py`, `classifier.py`, `db.py`, `requirements.txt`, `README.md`, `GUIDE.md`, `.gitignore`
   - the **`models`** folder (with `skin_model.onnx` + `labels.json`)
   - the **`training`** folder
3. **Commit changes**. The 16 MB model is fine. GitHub's web upload allows files up to 25 MB, and git pushes allow up to 100 MB.
   - Bigger model (e.g. ConvNeXt)? Upload it to **Hugging Face Hub** instead and ask for the app to download it at start-up.

### 5b. Database (optional, for permanent stats)
In TiDB Cloud → **SQL Editor**: `CREATE DATABASE skin;`

### 5c. Streamlit Community Cloud
1. **share.streamlit.io → Create app → Deploy a public app from GitHub**.
2. Repository `your-name/Skin-Check-AI`, branch `main`, main file `App.py`.
3. **Advanced settings → Python 3.13** and **Secrets**:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ADMIN_USER = "vanshika"
ADMIN_PASS = "a-strong-password"
# optional permanent storage:
MYSQL_HOST = "gateway01.us-east-1.prod.aws.tidbcloud.com"
MYSQL_PORT = "4000"
MYSQL_USER = "xxxxxxxx.root"
MYSQL_PASSWORD = "your-tidb-password"
MYSQL_DB = "skin"
MYSQL_SSL = "true"
```
4. **Deploy**. Test with a photo, then log in to **Admin**.

### Protecting your API bill
- No usage limits by default. Optionally set `MAX_ANALYSES_PER_SESSION` and/or `DAILY_ANALYSIS_LIMIT` in Secrets (0 = no limit).
- Set `APP_ACCESS_CODE = "something"` to require a code, e.g. for a demo to teachers only.
- Keep the spending limit in the Anthropic console.

---

## Part 6. Updating the model later
Retrain in Colab, then replace `models/skin_model.onnx` and `models/labels.json` on GitHub (**Add file → Upload files**). Streamlit redeploys automatically.

---

## Part 7. Limitations (say these in interviews)
- Public skin datasets are small and mostly lighter skin tones, so the model may be less accurate on darker skin.
- The model can only choose from the conditions it was trained on. That's why Claude double-checks it and the app shows both.
- Confidence is not certainty. This is an educational tool, not a diagnosis.
