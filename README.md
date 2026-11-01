# ai-resume-assistant
# 📄 ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and
suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Optional **job description** for a tailored keyword match
- Overall **ATS score (0–100)** computed from 5 weighted categories
  (Keywords, Experience & Impact, Formatting, Structure, Grammar)
- Strengths, prioritised improvements, missing keywords and bullet-point rewrites
- Downloadable JSON report

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at <https://aistudio.google.com/apikey>, then either:

**Option A – secrets file (recommended)**

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key"
# GEMINI_MODEL = "gemini-3.8-flash"   # optional override
```

**Option B – paste the key into the app's sidebar** when it runs.

Start the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (never commit `secrets.toml`).
2. Go to <https://share.streamlit.io> → **Create app** → select the repo, branch `main`, file `app.py`.
3. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-api-key"
   ```
4. Click **Deploy**.

## Notes
- Scanned/image-only PDFs can't be read (ATS systems can't read them either). Use a text-based PDF or DOCX.
- The score is an AI-based estimate, not the output of any specific commercial ATS.
- Resume text is sent to Google's Gemini API for analysis; the app itself stores nothing.
- Default model is `gemini-3.8-flash`. If it is ever retired, the app falls back to older Flash models automatically; you can also set `GEMINI_MODEL` to any current model name (see https://ai.google.dev/gemini-api/docs/models).

## Project structure
```
app.py            # Streamlit app
requirements.txt  # Python dependencies
README.md
```
