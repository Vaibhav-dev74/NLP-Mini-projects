# 💬 Dialogue Classifier 2.0 – Intelligent NLP-Based Text Classification System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-22%20passed-brightgreen.svg)]()
[![Flask](https://img.shields.io/badge/framework-Flask%203.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.9-orange.svg)](https://scikit-learn.org/)

A production-style Natural Language Processing (NLP) web platform that analyzes dialogues and sentences for **communicative intent (10 categories)**, **sentiment polarity**, **affective emotion**, **linguistic text statistics**, and **decision explainability**.

Supports **Rule-Based**, **Scikit-Learn Machine Learning (TF-IDF)**, and **Hybrid Consensus** classification with SQLite audit telemetry and batch CSV processing.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Supported Categories & Intents](#-supported-categories--intents)
- [Key Features](#-key-features)
- [Classification Modes](#-classification-modes)
- [Machine Learning Pipeline & Evaluation](#-machine-learning-pipeline--evaluation)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [REST API Documentation](#-rest-api-documentation)
- [Preserved Mini Projects](#-preserved-mini-projects)
- [Automated Testing](#-automated-testing)
- [Author](#-author)

---

## 🚀 Project Overview

The original **Dialogue Classifier v1.0** was a desktop Tkinter script using keyword matching across 6 dialogue categories. 

**Dialogue Classifier 2.0** elevates the project into an enterprise-ready NLP web application built with clean modular architecture, full explainability, and multi-modal intelligence:

1. **Intelligent Text Classification**: Single-sentence, multi-sentence, and paragraph dialogue classification.
2. **True Probabilities (No Fake Metrics)**: Authentic Scikit-Learn `predict_proba` probability distributions for ML/Hybrid models, and clear rule matches for rule-based models.
3. **Sentiment Analysis**: Positivity/negativity polarity scoring (-1.0 to +1.0) with negation and intensifier handling.
4. **Emotion Detection**: Affective state mapping into `HAPPY`, `SAD`, `ANGRY`, `FEAR`, `SURPRISE`, and `NEUTRAL`.
5. **Linguistic Preprocessing & Statistics**: Word count, character count, sentence count, reading time, and lexical richness.
6. **Batch CSV Processing**: Upload datasets, select target columns, process bulk records, and download enriched CSV files.
7. **SQLite Audit Trail & Analytics**: Real-time distribution charts (Chart.js) and search/filter/export capabilities (CSV & TXT).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    User([User / API Client]) --> Gateway[Flask REST API & Web UI / app.py]
    
    Gateway --> Preproc[src/preprocessing.py<br/>Tokenization, Normalization, Statistics]
    
    Preproc --> Router{Classification Mode}
    Router -->|Rule-Based Original| OrigRule[src/rule_classifier.py<br/>OriginalRuleClassifier (v1.0 Legacy)]
    Router -->|Rule-Based Enhanced| EnhRule[src/rule_classifier.py<br/>EnhancedRuleClassifier (10-Class, Regex \b)]
    Router -->|Machine Learning| ML[src/ml_classifier.py<br/>TF-IDF + Logistic Regression / Naive Bayes]
    Router -->|Hybrid Consensus| Hybrid[src/hybrid_classifier.py<br/>Rule Priorities + ML Probabilities]
    
    Preproc --> Sent[src/sentiment_analyzer.py<br/>Polarity, Negations, Intensifiers]
    Preproc --> Emo[src/emotion_detector.py<br/>6-Class Affective Marker Lexicon]
    
    OrigRule --> Aggregator[Result Synthesizer]
    EnhRule --> Aggregator
    ML --> Aggregator
    Hybrid --> Aggregator
    Sent --> Aggregator
    Emo --> Aggregator
    
    Aggregator --> DB[(SQLite Database<br/>instance/classifier.db)]
    Aggregator --> Response[Interactive SaaS Dashboard / JSON Response]
```

---

## 🎯 Supported Categories & Intents

The platform identifies 10 distinct communicative speech acts:

| Category | Description | Linguistic Pattern / Example |
| :--- | :--- | :--- |
| **GREETING** | Social opening salutation | *"Good morning team, hope you are doing well."* |
| **FAREWELL** | Departure and well-wishes | *"See you tomorrow, take care!"* |
| **QUESTION** | Inquiries seeking information | *"Where are the deployment logs stored?"* |
| **REQUEST** | Polite directives asking for action | *"Could you please review the pull request?"* |
| **EXCLAMATION** | High-intensity emotional interjection | *"Wow, that was an absolutely incredible result!"* |
| **STATEMENT** | Informative declaration of facts | *"The quarterly metrics are uploaded on the server."* |
| **COMPLAINT** | Dissatisfaction or bug reports | *"This app keeps crashing and is completely broken."* |
| **SUGGESTION** | Recommendations and advice | *"You should try clearing your browser cache."* |
| **APOLOGY** | Expressions of regret or excuses | *"I am truly sorry for the delayed response."* |
| **THANKS** | Gratitude and appreciation | *"Thank you so much for your guidance and help!"* |

---

## ⚙️ Classification Modes

1. **Hybrid Classification (Recommended)**
   - Synthesizes rule-based precision with probabilistic machine learning.
   - Reaches consensus when ML and rules agree.
   - Prioritizes explicit rule patterns (such as polite requests, apologies, greetings) when ML confidence is ambiguous, and defers to ML when rule defaults to a plain statement.
   - Fully explains decision provenance.

2. **Enhanced Rule-Based Engine (v2.0)**
   - Utilizes strict regex word boundaries (`\b`) preventing substring false positives (e.g. `'hi'` in `'history'`).
   - Supports multi-sentence dialogues, ranking sentences by communicative hierarchy.
   - Extracts matched keywords and syntactic patterns.

3. **Original Rule-Based Engine (v1.0 Legacy)**
   - 100% faithful backward-compatible preservation of the original Tkinter algorithm from `dialogue_classifier.py`.

4. **Machine Learning (TF-IDF + Scikit-Learn)**
   - Unigram and bigram TF-IDF vectorization.
   - True probability distributions via `predict_proba` with confidence margins over secondary classes.

---

## 📊 Machine Learning Pipeline & Evaluation

The machine learning pipeline implements:
1. Dataset ingestion and validation (`data/dialogue_dataset.csv`).
2. Train/Test split with stratification (80% train, 20% test).
3. TF-IDF vectorizer fit **only on training split** to strictly prevent data leakage.
4. Dual model training and evaluation:
   - **Logistic Regression** (with balanced class weights)
   - **Multinomial Naive Bayes** (with laplace smoothing)
5. Comparison of Accuracy, Weighted Precision, Weighted Recall, and Weighted F1-Score.
6. Automated champion model selection and serialization using `joblib`.

### Benchmark Results on Demonstration Dataset:

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **83.33%** | **87.50%** | **83.33%** | **82.38%** | 🏆 **Champion Model** |
| **Multinomial Naive Bayes** | 80.00% | 83.33% | 80.00% | 78.57% | Benchmarked |

*Note: Results obtained on the baseline 150-sample curated dialogue dataset. Demonstrates functional ML pipeline, feature extraction, and model comparison.*

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Flask 3.1, Werkzeug
- **Machine Learning & NLP**: Scikit-Learn 1.9, TF-IDF Vectorizer, Joblib, NumPy, Pandas
- **Database**: SQLite3 (Thread-safe, parameterized queries, indexed)
- **Frontend**: HTML5, CSS3 (Modern SaaS Design System, Dark/Light Themes), Vanilla JavaScript (ES6+)
- **Visualization**: Chart.js 4.x
- **Testing**: Pytest 9.2 (22 unit & integration tests)

---

## 📁 Project Structure

```text
NLP-Mini-projects/
│
├── dialogue_classifier.py          # Preserved v1.0 Tkinter desktop app (untouched)
├── Emoji-text.py                  # Preserved bidirectional emoji converter (untouched)
├── app.py                         # Flask web app & REST API gateway
├── config.py                      # Application configuration & paths
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
│
├── data/
│   ├── dialogue_dataset.csv       # Labeled training dataset (10 categories)
│   └── uploads/                   # Batch CSV processing folder
│
├── models/
│   ├── classifier_model.pkl       # Serialized champion model (Joblib)
│   ├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer (Joblib)
│   └── model_metadata.json        # Evaluation metrics & comparison report
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── preprocessing.py           # Text normalization, tokenization & statistics
│   ├── rule_classifier.py         # Original v1.0 & Enhanced v2.0 rule engines
│   ├── ml_classifier.py           # Scikit-Learn TF-IDF training & inference
│   ├── hybrid_classifier.py       # Rule-ML consensus & provenance engine
│   ├── sentiment_analyzer.py      # Lexicon-based sentiment & polarity
│   ├── emotion_detector.py        # 6-class affective emotion detector
│   ├── database.py                # SQLite management & history CRUD
│   ├── analytics.py               # Distribution & timeline metric aggregator
│   └── batch_processor.py         # CSV validation & bulk classification
│
├── templates/
│   ├── base.html                  # Base dashboard layout with sidebar
│   ├── dashboard.html             # Platform overview & quick classifier
│   ├── classifier.html            # Full text classification workspace
│   ├── batch.html                 # CSV batch processing & download
│   ├── history.html               # Audit history, search & CSV/TXT exports
│   ├── analytics.html             # Interactive Chart.js analytics charts
│   └── about.html                 # Technical specs & model retrainer
│
├── static/
│   ├── css/style.css              # Light & Dark theme responsive styling
│   └── js/app.js                  # Client controllers & Chart.js rendering
│
└── tests/
    ├── test_rule_classifier.py    # Rule engine tests (legacy & enhanced)
    ├── test_preprocessing.py      # Preprocessing & statistics tests
    ├── test_ml_classifier.py      # ML training, evaluation & hybrid tests
    ├── test_sentiment_emotion.py  # Sentiment & emotion tests
    ├── test_database_and_batch.py # SQLite CRUD & batch processor tests
    └── test_api.py                # REST API integration tests
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Vaibhav-dev74/NLP-Mini-projects.git
cd NLP-Mini-projects
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option A: Run Dialogue Classifier 2.0 (Web Platform)
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### Option B: Run Legacy Dialogue Classifier (Tkinter GUI)
```bash
python dialogue_classifier.py
```

### Option C: Run Emoji Converter (Tkinter GUI)
```bash
python Emoji-text.py
```

---

## 🌐 REST API Documentation

### 1. Single Text Classification
- **Endpoint**: `POST /api/classify`
- **Request Body**:
  ```json
  {
    "text": "Hello team! Could you please send me the final report?",
    "mode": "hybrid",
    "save_history": true
  }
  ```
- **Response**:
  ```json
  {
    "category": "REQUEST",
    "confidence": 0.88,
    "confidence_score": 0.88,
    "mode": "hybrid",
    "subsystem_used": "hybrid_consensus",
    "explanation": "Consensus reached: Both the Enhanced Rule Engine and Logistic Regression agreed on 'REQUEST'.",
    "matched_keywords": ["please", "send", "report"],
    "sentiment": {
      "sentiment": "POSITIVE",
      "polarity_score": 0.45,
      "intensity": "Moderate"
    },
    "emotion": {
      "emotion": "HAPPY",
      "confidence": 0.75
    },
    "statistics": {
      "word_count": 9,
      "character_count": 55,
      "sentence_count": 2,
      "reading_time_seconds": 2.57
    }
  }
  ```

### 2. Batch CSV Classification
- **Inspect CSV**: `POST /api/batch/inspect` (multipart/form-data with `file`)
- **Execute Batch**: `POST /api/classify/batch` (`temp_filename`, `text_column`, `mode`)

### 3. Classification History
- **Get History**: `GET /api/history?page=1&limit=20&category=ALL&sentiment=ALL&search=`
- **Delete Record**: `DELETE /api/history/<id>`
- **Clear All**: `POST /api/history/clear`

### 4. Telemetry & Export
- **Analytics Metrics**: `GET /api/analytics`
- **Export CSV**: `GET /api/export/csv`
- **Export Summary Report**: `GET /api/export/report`
- **System Health**: `GET /api/health`
- **Retrain Models**: `POST /api/train`

---

## 🧪 Automated Testing

The project includes 22 automated unit and integration tests covering preprocessing, rule precision, ML training, probabilities, database CRUD, sentiment, emotion, and all REST endpoints.

Run the test suite:
```bash
pytest -v
```

Output:
```text
tests/test_api.py::test_api_health PASSED
tests/test_api.py::test_api_classify_empty PASSED
tests/test_api.py::test_api_classify_modes PASSED
tests/test_api.py::test_api_history_and_analytics PASSED
tests/test_api.py::test_api_exports PASSED
tests/test_api.py::test_api_batch_inspect_and_process PASSED
tests/test_database_and_batch.py::test_database_crud PASSED
tests/test_database_and_batch.py::test_analytics_metrics PASSED
tests/test_ml_classifier.py::test_ml_training_and_evaluation PASSED
tests/test_ml_classifier.py::test_ml_prediction_probabilities PASSED
tests/test_ml_classifier.py::test_hybrid_classification PASSED
tests/test_preprocessing.py::test_empty_input PASSED
tests/test_preprocessing.py::test_whitespace_and_unicode_normalization PASSED
tests/test_preprocessing.py::test_key_punctuation_preservation PASSED
tests/test_preprocessing.py::test_sentence_splitting PASSED
tests/test_preprocessing.py::test_text_statistics PASSED
tests/test_rule_classifier.py::test_legacy_original_classifier_compatibility PASSED
tests/test_rule_classifier.py::test_enhanced_classifier_10_categories PASSED
tests/test_rule_classifier.py::test_word_boundary_precision PASSED
tests/test_rule_classifier.py::test_multi_sentence_handling PASSED
tests/test_sentiment_emotion.py::test_sentiment_positive_negative_neutral PASSED
tests/test_sentiment_emotion.py::test_emotion_detection PASSED

============================= 22 passed in 5.14s ==============================
```

---

## 👨‍💻 Author

**Vaibhav Pratap Mishra**

- GitHub: [@Vaibhav-dev74](https://github.com/Vaibhav-dev74)
- Repository: [NLP-Mini-projects](https://github.com/Vaibhav-dev74/NLP-Mini-projects.git)
