

from flask import Flask, render_template, request
import PyPDF2
import sqlite3
import spacy
nlp = spacy.load("en_core_web_sm")
from datetime import datetime

app = Flask(__name__)

# -------------------------------
# Skill Database
# -------------------------------
skills_db = [
    "python", "java", "c++", "sql",
    "flask", "django",
    "machine learning", "data science",
    "html", "css", "javascript",
    "react", "nodejs",
    "aws", "azure", "docker",
    "git", "linux"
]

# -------------------------------
# Initialize Database
# -------------------------------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            score REAL,
            matched TEXT,
            missing TEXT,
            date TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# -------------------------------
# Similarity Calculation
# -------------------------------
def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors[0], vectors[1])

    # NLP Processing
    doc_resume = nlp(resume_text.lower())
    doc_job = nlp(job_desc.lower())

    resume_words = set([token.lemma_ for token in doc_resume 
                        if not token.is_stop and token.is_alpha])

    job_words = set([token.lemma_ for token in doc_job 
                     if not token.is_stop and token.is_alpha])

    matched_skills = resume_words.intersection(job_words)
    missing_skills = job_words.difference(resume_words)

    return round(float(similarity[0][0]) * 100, 2), matched_skills, missing_skills



# -------------------------------
# Main Route
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    ranked_results = []

    if request.method == 'POST':
        files = request.files.getlist("resume")
        job_desc = request.form['job_desc']

        for file in files:
            if file and file.filename.endswith(".pdf"):

                pdf_reader = PyPDF2.PdfReader(file)
                resume_text = ""

                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        resume_text += text

                score, matched, missing = calculate_similarity(resume_text, job_desc)

                # Save to database
                conn = sqlite3.connect("database.db")
                c = conn.cursor()

                c.execute("""
                    INSERT INTO results (filename, score, matched, missing, date)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    file.filename,
                    score,
                    str(matched),
                    str(missing),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))

                conn.commit()
                conn.close()

        # Fetch ranked results (highest score first)
        conn = sqlite3.connect("database.db")
        c = conn.cursor()

        c.execute("SELECT filename, score FROM results ORDER BY score DESC")
        ranked_results = c.fetchall()

        conn.close()

    return render_template("index.html", ranked_results=ranked_results)


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

