"""
Ankur Foundation - Analysis API
=================================
A single Flask app that serves BOTH features on one port:

  GET  /                -> health check
  POST /analyze-resume  -> AI resume analysis (OpenRouter)
  GET  /next-6-months   -> donation forecast (Prophet model)

Run locally:
    python app.py            # http://localhost:5000

Deploy on Render (start command):
    gunicorn app:app
"""

import os
import re
import json
import pickle
import threading

import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import PyPDF2
import docx

# -----------------------------------------------------------------------------
# CONFIG / ENV
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from the analysis root, falling back to the old resume_api/.env so
# nothing breaks for an existing local setup.
for candidate in (os.path.join(BASE_DIR, ".env"),
                  os.path.join(BASE_DIR, "resume_api", ".env")):
    if os.path.exists(candidate):
        load_dotenv(candidate)
        break

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# The Prophet model is checked into the repo. Look in a few likely spots so the
# app works whether it's launched from analysis/ or analysis/api/.
def _find_model():
    for p in (os.path.join(BASE_DIR, "donation_prophet_model.pkl"),
              os.path.join(BASE_DIR, "model", "donation_prophet_model.pkl"),
              os.path.join(BASE_DIR, "api", "donation_prophet_model.pkl")):
        if os.path.exists(p):
            return p
    return None

MODEL_PATH = _find_model()

# In-memory model — retraining updates this without touching disk
_prophet_model = None
_model_lock = threading.Lock()

# InfinityFree DB — used for live retraining on Render
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",   "sql101.infinityfree.com"),
    "user":     os.getenv("DB_USER",   "if0_41513499"),
    "password": os.getenv("DB_PASS",   "ankurvelu2026"),
    "database": os.getenv("DB_NAME",   "if0_41513499_ankur"),
    "connect_timeout": 30,
}

def _load_model_from_file():
    """Load the bundled .pkl into memory at startup."""
    global _prophet_model
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            with _model_lock:
                _prophet_model = pickle.load(f)
        print("Model loaded from file:", MODEL_PATH)
    else:
        print("No .pkl found — will retrain from DB on first /retrain call.")

def _retrain_from_db():
    """Fetch live donation data from InfinityFree and retrain Prophet in memory."""
    try:
        import pymysql
        conn = pymysql.connect(**DB_CONFIG)
        df = pd.read_sql(
            "SELECT DonationDate as ds, Amount as y FROM donar_donations "
            "WHERE DonationDate IS NOT NULL AND Amount > 0 ORDER BY DonationDate",
            conn
        )
        conn.close()

        if df.empty:
            return False, "No donation data in DB"

        from prophet import Prophet
        df["ds"] = pd.to_datetime(df["ds"]).dt.to_period("M").dt.to_timestamp()
        df["y"]  = df["y"].astype(float)
        df = df.groupby("ds", as_index=False)["y"].sum()

        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(df)

        global _prophet_model
        with _model_lock:
            _prophet_model = model

        print(f"Model retrained on {len(df)} monthly records.")
        return True, f"Retrained on {len(df)} monthly records"

    except Exception as e:
        print("Retrain error:", e)
        return False, str(e)

app = Flask(__name__)
CORS(app)

# Load bundled model at startup, then kick off a background retrain from live DB
_load_model_from_file()

def _background_retrain():
    ok, msg = _retrain_from_db()
    print("Background retrain:", msg)

threading.Thread(target=_background_retrain, daemon=True).start()


# -----------------------------------------------------------------------------
# RESUME: TEXT EXTRACTION  (no OCR — text-based PDF/DOCX only)
# -----------------------------------------------------------------------------
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print("PDF read error:", e)
    return text


def extract_text_from_docx(file_path):
    text = ""
    try:
        document = docx.Document(file_path)
        for para in document.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print("DOCX read error:", e)
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    words = list(dict.fromkeys(words))   # de-dupe while keeping order
    return " ".join(words)


# -----------------------------------------------------------------------------
# RESUME: AI ANALYSIS (OpenRouter)
# -----------------------------------------------------------------------------
def analyze_with_ai(text):
    prompt = f"""
You are a professional HR Resume Analyzer.

Analyze this resume and return STRICT JSON only.

Resume:
{text[:2000]}

Format:
{{
  "domain": "",
  "score": 0,
  "skills": [],
  "best_roles": [],
  "internships": [],
  "skill_gaps": [],
  "technologies_to_learn": [],
  "feedback": []
}}
"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct",  # free model
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Strip markdown fences and isolate the JSON object.
        content = content.replace("```json", "").replace("```", "")
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end != 0:
            content = content[start:end]

        try:
            result = json.loads(content)
        except Exception as e:
            print("JSON parse error:", e, "| content:", content)
            return {"error": "Invalid AI response"}

        # Attach job-search links for each recommended role.
        job_links = {}
        for role in result.get("best_roles", []):
            role_slug = str(role).replace(" ", "-").lower()
            role_url = str(role).replace(" ", "%20")
            role_plus = str(role).replace(" ", "+")
            job_links[role] = {
                "naukri": f"https://www.naukri.com/{role_slug}-jobs",
                "linkedin": f"https://www.linkedin.com/jobs/search/?keywords={role_url}",
                "indeed": f"https://www.indeed.com/jobs?q={role_plus}",
            }
        result["job_links"] = job_links
        return result

    except Exception as e:
        print("OpenRouter error:", e)
        return None


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "ankur-analysis-api",
        "openrouter_key_loaded": bool(OPENROUTER_API_KEY),
        "forecast_model_loaded": MODEL_PATH is not None,
    })


@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    if not OPENROUTER_API_KEY:
        return jsonify({"error": "OpenRouter API key missing"}), 500

    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    temp_path = os.path.join(BASE_DIR, f"temp_{os.getpid()}.{ext}")
    file.save(temp_path)

    try:
        if ext == "pdf":
            text = extract_text_from_pdf(temp_path)
        elif ext == "docx":
            text = extract_text_from_docx(temp_path)
        else:
            return jsonify({"error": "Only PDF/DOCX allowed"}), 400
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not text.strip():
        return jsonify({"error": "Text extraction failed"}), 400

    result = analyze_with_ai(clean_text(text))
    if not result:
        return jsonify({"error": "AI failed"}), 500

    return jsonify(result)


@app.route("/next-6-months", methods=["GET"])
def forecast_next_6_months():
    with _model_lock:
        model = _prophet_model

    if model is None:
        return jsonify({"error": "Model not ready yet — retrain in progress"}), 503

    future = model.make_future_dataframe(periods=6, freq="M")
    forecast = model.predict(future)
    rows = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6)

    output = []
    for _, row in rows.iterrows():
        output.append({
            "Month": row["ds"].strftime("%B"),
            "Year": int(row["ds"].year),
            "Predicted_Donation": round(row["yhat"], 2),
            "Min_Estimate": round(row["yhat_lower"], 2),
            "Max_Estimate": round(row["yhat_upper"], 2),
        })
    return jsonify(output)


@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    """Manually trigger a retrain from live DB. Call this from a cron job."""
    secret = os.getenv("RETRAIN_SECRET", "")
    if secret and request.args.get("secret") != secret:
        return jsonify({"error": "Unauthorized"}), 401

    ok, msg = _retrain_from_db()
    return jsonify({"success": ok, "message": msg}), (200 if ok else 500)


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
