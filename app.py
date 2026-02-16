
from flask import Flask, render_template, request
import os
import sqlite3
import PyPDF2
import spacy
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------
# App Setup
# --------------------------------
app = Flask(__name__)

# --------------------------------
# Load spaCy Model
# --------------------------------
nlp = spacy.load("en_core_web_sm")

# --------------------------------
# Database Setup
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS resumes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    score REAL,
    timestamp TEXT
)
""")
conn.commit()

# --------------------------------
# Extract Text From PDF
# --------------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# --------------------------------
# Calculate Similarity
# --------------------------------
def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors[0], vectors[1])

    # NLP cleaning
    doc_resume = nlp(resume_text.lower())
    doc_job = nlp(job_desc.lower())

    resume_words = set([token.lemma_ for token in doc_resume 
                        if not token.is_stop and token.is_alpha])

    job_words = set([token.lemma_ for token in doc_job 
                     if not token.is_stop and token.is_alpha])

    matched_skills = resume_words.intersection(job_words)
    missing_skills = job_words.difference(resume_words)

    score = round(float(similarity[0][0]) * 100, 2)

    return score, matched_skills, missing_skills

# --------------------------------
# Main Route
# --------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        files = request.files.getlist("resume")
        job_desc = request.form["job_desc"]

        results = []

        for file in files:
            if file and file.filename.endswith(".pdf"):

                resume_text = extract_text_from_pdf(file)
                score, matched, missing = calculate_similarity(resume_text, job_desc)

                # Save to DB
                c.execute(
                    "INSERT INTO resumes (filename, score, timestamp) VALUES (?, ?, ?)",
                    (file.filename, score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                conn.commit()

                results.append({
                    "filename": file.filename,
                    "score": score,
                    "matched": matched,
                    "missing": missing
                })

        # Sort highest score first
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return render_template("index.html", results=results)

    return render_template("index.html")


# --------------------------------
# Run App
# --------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


