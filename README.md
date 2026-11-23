
# Automatic CV Evaluator and Job Matcher 

An end-to-end NLP-powered application that evaluates resumes against an industry-standard scoring rubric, maps candidate profiles to job descriptions using multiple state-of-the-art transformer models, performs semantic skill-gap analysis, and hosts an interactive web dashboard via **Gradio**.

## Project Overview

Finding the right candidate or matching your resume to a dream job is a tedious process. This project automates CV evaluation and job description matching using a hybrid approach combining **lexical rules**, **custom NER (Named Entity Recognition)** with **spaCy**, and **Semantic Search** via Deep Learning sentence embeddings.

### Core Features
- **Comprehensive 9-Category CV Rubric Scoring:** Evaluates resumes across 9 distinct criteria (Academic Record, Work Experience, Projects, Technical Skills, Soft Skills, Languages, Certifications, Achievements/Leadership, and Publications) for a total score of 100.
- **Multi-Model Semantic Matching:** Uses 3 state-of-the-art sentence transformer models alongside a classic TF-IDF baseline:
  - `Sentence-BERT (all-MiniLM-L6-v2)`
  - `RoBERTa Large v1 (all-roberta-large-v1)`
  - `DistilRoBERTa (all-distilroberta-v1)`
  - `TF-IDF Baseline (Lexical benchmark).`
- **Skill Gap & Learning Roadmap Generator:** Highlights exactly what core skills are missing compared to the job description and suggests a customized learning roadmap.
- **Explainable AI (XAI) Summary:** Explains how scores were computed and why a specific transformer model was selected as the best match.
- **Gradio Interactive UI:** An elegant, fully functional web interface where users can upload PDFs (using **PyMuPDF** & **Tesseract OCR** for scanned documents) and immediately view real-time analysis.

## System Architecture & Workflow

1. **Data Ingestion & Cleaning:** Consolidated resume and job datasets (e.g., ESCO taxonomy, fake job postings) to prepare robust training and baseline data.
2. **Custom Entity Extraction:** Utilized **spaCy** phrase matchers and rule-based keyword matching to extract skills, tools, roles, domains, and certifications.
3. **Hybrid Feature Engineering:** Parsed PDF structures, extracted formal experience years, academic achievements, and structured them into JSON representations.
4. **Model Training & Comparison:** Computed dense embeddings of resumes and jobs, calculated cosine similarity matrices, and ranked model performance against manual human annotations.
5. **Deployment:** Structured the backend pipeline to integrate seamlessly into a clean multi-tab Gradio UI.

---

## Installation & Setup

To run this project locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/automatic-cv-evaluator.git
cd automatic-cv-evaluator
pip install -r requirements.txt
```

### Prerequisites
You also need to install system dependencies for PDF processing and OCR:
```bash
# Linux / Ubuntu
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
```

### `requirements.txt`
Make sure your requirements include:
```text
gradio
pymupdf
pytesseract
pillow
sentence-transformers
spacy
scikit-learn
pandas
numpy
scipy
joblib
```
Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Evaluation Metrics

We evaluated our AI system's matching capabilities against ground-truth human annotations:
- **Accuracy:** ~55%
- **Precision:** ~67.2%
- **Recall:** ~60%
- **F1-Score:** ~63.4%

Our analysis showed that **Sentence-RoBERTa** and **Sentence-BERT** consistently capture deep semantic matching far better than traditional keyword-based **TF-IDF Baseline** pipelines.

## Gradio Web UI Dashboard

The web app contains several diagnostic tabs:
- **Summary:** High-level compatibility summary and extraction health parameters.
- **ATS Score:** Quick visualization of applicant tracking system scores.
- **9-Category Score:** Granular breakdowns with progress statuses (Strong/Average/Weak).
- **Skills & Keyword Analytics:** Side-by-side display of technical skills, soft skills, and natural frequency phrases.
- **Job Match & Gap Analysis:** Direct comparison of skill coverage, missing requisites, and estimated versus required professional years.
- **Learning Plan:** Dynamically generated roadmap checklist to cover identified technical deficits.
- **Model Dashboard:** Technical performance breakdown of all 4 tested NLP architectures.

## Contribution
Contributions, issues, and feature requests are welcome!
