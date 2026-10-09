# ai-resume-assistant
# 📄 ATS Resume Checker

A Streamlit app that scores a resume for Applicant Tracking System (ATS) compatibility and
suggests concrete improvements, powered by Google's Gemini Flash model.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- Optionally paste a **job description** for targeted keyword feedback
- **Estimated ATS score (0-100)** with a weighted category breakdown
  (formatting, keywords, impact, skills, structure)
- Strengths, missing keywords, prioritised improvements and example bullet rewrites
- Downloadable JSON report

> ⚠️ The score is an AI-based **estimate**. Real ATS products behave differently from each
> other, so treat it as guidance, not a guarantee. Resume text is sent to the Gemini API.

## Project structure

```
.
├── app.py            # Streamlit app
├── requirements.txt  # Python dependencies
├── README.md
└── .gitignore
```

## Run locally

1. Install Python 3.9+ and get a free Gemini API key at <https://aistudio.google.com/apikey>.
2. Set up and run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your-key"  # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

You can also skip the environment variable and paste the key into the sidebar field.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | env var, or Streamlit secret | *(required)* |
| `GEMINI_MODEL` | env var, Streamlit secret, or sidebar field | `gemini-3.5-flash` |

Google renames and retires models over time. If you get a "model not found" error, check the
current Flash model name at <https://ai.google.dev/gemini-api/docs/models> and change it in
the sidebar or via `GEMINI_MODEL`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (keep `.gitignore`, never commit your API key).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → choose your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Limitations

- Scanned/image-only PDFs have no extractable text (and ATS systems can't read them either).
- Resume text is truncated to 20,000 characters and the job description to 8,000.
- Free Gemini API tiers have rate limits; you may see a quota error under heavy use.
