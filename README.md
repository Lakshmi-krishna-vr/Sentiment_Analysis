# 🧠 Sentiment Analysis of Amazon Reviews (VADER + RoBERTa)

## 📌 Project Overview

This project performs **Sentiment Analysis on Amazon product reviews** using two approaches:

* 🟢 **VADER (NLTK)** – Rule-based sentiment analysis
* 🤖 **RoBERTa (HuggingFace Transformers)** – Deep learning-based sentiment model

The goal is to **analyze review text**, compare it with **star ratings**, and evaluate how well sentiment aligns with user feedback.

---

## 🎯 Objectives

* Analyze customer reviews and classify sentiment as:

  * Positive 😊
  * Neutral 😐
  * Negative 😠
* Compare:

  * ⭐ Star Ratings (`Score`)
  * 💬 Text Sentiment
* Evaluate differences between:

  * Rule-based vs Deep Learning models

---

## 📂 Dataset

* Amazon Reviews Dataset (CSV format)
* Contains:

  * Review Text
  * Score (1–5 stars)
  * Other metadata

> ⚠️ Note: Only the first **500 reviews** are used for faster processing.

---

## ⚙️ Technologies Used

### 🐍 Programming Language

* Python

### 📚 Libraries

* `pandas` – Data handling
* `numpy` – Numerical operations
* `matplotlib`, `seaborn` – Visualization
* `nltk` – NLP preprocessing + VADER
* `transformers` – RoBERTa model
* `scipy` – Softmax function
* `tqdm` – Progress bar

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Exploration

* Loaded dataset using pandas
* Checked shape and preview
* Visualized distribution of review scores

### 2️⃣ Data Visualization

* Bar chart of review counts by star rating
* Observed skew toward positive reviews

### 3️⃣ Sentiment Analysis (VADER)

* Used `SentimentIntensityAnalyzer`
* Generated:

  * `neg`, `neu`, `pos`, `compound` scores
* Stored results in DataFrame

### 4️⃣ Visualization of VADER Results

* Compared sentiment score vs star ratings
* Found strong correlation

### 5️⃣ Sentiment Analysis (RoBERTa)

* Used pretrained model:

  ```
  cardiffnlp/twitter-roberta-base-sentiment
  ```
* Tokenized text and computed probabilities
* Extracted:

  * Negative
  * Neutral
  * Positive scores

### 6️⃣ Model Combination

* Combined VADER + RoBERTa outputs
* Created unified results dataset

### 7️⃣ Full Processing

* Iterated through all reviews using `tqdm`
* Applied both models
* Handled runtime errors

### 8️⃣ Advanced Visualization

* Pairplot to compare:

  * VADER vs RoBERTa
  * Sentiment vs Ratings

### 9️⃣ HuggingFace Pipeline Test

* Used:

  ```python
  pipeline("sentiment-analysis")
  ```
* Tested model on sample sentences

---

## 📊 Results & Insights

* ⭐ Reviews are heavily skewed toward **5-star ratings**
* 📈 Strong correlation between:

  * Star ratings and sentiment scores
* 🤖 RoBERTa provides:

  * Better context understanding
  * More accurate predictions
* 🟢 VADER is:

  * Faster
  * Simpler but less accurate

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn nltk transformers scipy tqdm
```

### 2️⃣ Download NLTK Resources

```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

### 3️⃣ Run the Notebook / Script

* Open in Google Colab or Jupyter Notebook
* Execute all cells step-by-step

---

## 📁 Project Structure

```
├── Reviews.csv.zip
├── sentiment_analysis.ipynb
├── README.md
```

---

## 💡 Key Concepts Learned

* Natural Language Processing (NLP)
* Sentiment Analysis Techniques
* Lexicon-based vs Transformer models
* Tokenization & Softmax
* Data Visualization
* Model comparison

---

## 🔮 Future Improvements

* Add accuracy metrics (F1-score, confusion matrix)
* Train custom sentiment model
* Deploy as web app (Streamlit / Flask)
* Real-time sentiment dashboard

---

## 👩‍💻 Author

**Lakshmi Krishna V R**

---

## ⭐ Acknowledgements

* NLTK for VADER
* HuggingFace for Transformers
* Amazon Reviews Dataset

---

## 📌 Short Summary

> This project analyzes Amazon reviews using both rule-based (VADER) and deep learning (RoBERTa) models, comparing sentiment scores with star ratings to evaluate performance and insights.

---
