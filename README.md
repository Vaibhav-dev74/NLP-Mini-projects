# 🧠 NLP Mini Projects

A collection of beginner-friendly **Natural Language Processing (NLP)** mini projects developed using **Python and Tkinter**. These projects demonstrate fundamental NLP concepts such as text preprocessing, rule-based classification, keyword matching, punctuation analysis, dictionary mapping, and phrase matching.

---

## 📌 Projects

### 💬 1. Dialogue Classifier

A rule-based NLP application that analyzes a user's sentence and classifies it into different dialogue categories.

#### Supported Categories

* 👋 **Greeting**
* 👋 **Farewell**
* ❓ **Question**
* 🙏 **Request**
* ❗ **Exclamation / Emphasis**
* 📝 **Statement**
* ⚠️ **Empty Input**

#### How It Works

The classifier processes the input text using:

* Text cleaning and normalization
* Lowercase conversion
* Punctuation analysis
* Keyword matching
* Phrase matching
* Rule-based classification

For example:

```text
"Hello there"       → GREETING
"Where are you?"    → QUESTION
"Please help me"    → REQUEST
"Goodbye!"          → EXCLAMATION/EMPHASIS
"I am learning NLP" → STATEMENT
```

---

### 😀 2. Bi-Directional Emoji Converter

A Tkinter-based application that converts **emojis into their textual meanings** and **words/phrases into emojis**.

#### Features

* 😀 Emoji → Word conversion
* 📝 Word → Emoji conversion
* 🔤 Case-insensitive word matching
* 🔗 Multi-word phrase matching
* 📋 Displays conversion results
* ⚠️ Handles unknown words/emojis
* 🖥️ Interactive Tkinter GUI

Example:

```text
Emoji → Word

😀 → Grinning Face
😂 → Tears of Joy
🔥 → Fire
❤️ → Red Heart
🚀 → Rocket / Launch
```

```text
Word → Emoji

fire → 🔥
pizza → 🍕
laptop → 💻
rocket → 🚀
thumbs up → 👍
```

The converter uses a custom emoji dictionary and creates a reverse mapping for word-to-emoji conversion.

---

## 🛠️ Technologies Used

* **Python 3**
* **Tkinter**
* **Natural Language Processing Concepts**
* **String Processing**
* **Dictionary / Hash Mapping**
* **Keyword Matching**
* **Rule-Based Classification**
* **Phrase Matching**

---

## 📂 Project Structure

```text
NLP-Mini-projects/
│
├── dialogue_classifier.py
├── Emoji-text.py
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Vaibhav-dev74/NLP-Mini-projects.git
```

### 2. Navigate to the Project

```bash
cd NLP-Mini-projects
```

### 3. Run the Dialogue Classifier

```bash
python dialogue_classifier.py
```

### 4. Run the Emoji Converter

```bash
python Emoji-text.py
```

---

## 🎯 Learning Objectives

These mini projects were created to understand the fundamentals of NLP and Python GUI development.

Key concepts demonstrated:

* Text preprocessing
* Text normalization
* Keyword-based NLP
* Rule-based text classification
* Punctuation-based analysis
* Dictionary-based mapping
* Reverse dictionary mapping
* Multi-word phrase detection
* GUI development using Tkinter

---

## 🔮 Future Improvements

Potential improvements include:

* 🤖 Machine Learning-based dialogue classification
* 🧠 AI/LLM-based text classification
* 😊 Sentiment analysis
* 🌐 Multilingual NLP support
* 🎤 Voice input
* 📚 Larger emoji vocabulary
* 🔍 Context-aware classification
* 🌐 Web-based version using Flask or React
* 📊 NLP analytics dashboard

---

## 👨‍💻 Author

**Vaibhav Pratap Mishra**

Computer Engineering Graduate

**Skills:** Python | NLP | AI | Full-Stack Development | IoT

🔗 GitHub: [Vaibhav-dev74](https://github.com/Vaibhav-dev74)

