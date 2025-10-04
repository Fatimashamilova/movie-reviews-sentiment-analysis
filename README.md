# 🎬 Movie Reviews Sentiment Analysis 

## 🧠 Overview

This project applies **Natural Language Processing (NLP)** and **Machine Learning** techniques to classify movie reviews as **positive** or **negative** based on text sentiment.  
The analysis was conducted using **Python**, leveraging popular NLP libraries to preprocess text data, extract meaningful features, and train classification models.

The project demonstrates end-to-end data mining — from data preprocessing and feature extraction to model training, evaluation, and visualization.

---

## 🎯 Objectives

- Perform **text preprocessing** including tokenization, stopword removal, and stemming.
- Apply **feature extraction** using **Bag-of-Words (BoW)** and **TF-IDF** vectorization.
- Train and evaluate **Naïve Bayes** and **K-Nearest Neighbors (KNN)** classifiers.
- Compare model performance and visualize sentiment distribution across the dataset.
- Demonstrate NLP pipeline design and interpretation using Python and scikit-learn.

---

## 📦 Dataset

- **Source:** [IMDB Movie Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
  or NLTK’s built-in *movie_reviews* corpus for smaller-scale experiments.  
- **Records:** 50,000 labeled reviews (positive / negative).  
- **Target Variable:** Sentiment (`positive`, `negative`).  
- **Data Fields:** Review Text, Sentiment Label.

---

## ⚙️ Tools and Libraries

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | **Python 3.10+** |
| IDE / Notebook | **Jupyter Notebook** |
| NLP & Text Mining | **NLTK**, **scikit-learn**, **re** |
| Data Processing | **Pandas**, **NumPy** |
| Visualization | **Matplotlib**, **WordCloud**, **Seaborn** |
| Machine Learning | **Multinomial Naïve Bayes**, **K-Nearest Neighbors (KNN)** |

---

## 🧩 Project Workflow

### 1️⃣ Data Loading
Load and preview the dataset to understand review structure and label balance.

### 2️⃣ Text Preprocessing
- Converted all text to lowercase.  
- Removed punctuation, numbers, and special characters.  
- Tokenized words and removed English stopwords using **NLTK**.  
- Applied **stemming** to reduce words to their base forms.

### 3️⃣ Feature Extraction
- Created feature matrices using:
  - **Bag-of-Words (BoW)**
  - **TF-IDF (Term Frequency–Inverse Document Frequency)**

### 4️⃣ Model Building
Trained two supervised classifiers:
- **Naïve Bayes (MultinomialNB)**
- **K-Nearest Neighbors (KNN)**

### 5️⃣ Model Evaluation
- Evaluated models using:
  - Accuracy
  - Precision, Recall, F1-Score
  - Confusion Matrix visualization  
- Compared both models to identify the best-performing algorithm.

### 6️⃣ Visualization
Generated **WordClouds** for positive and negative reviews and plotted sentiment distribution using **Matplotlib** and **Seaborn**.

---

## 📈 Results

| Model | Accuracy | F1-Score | Key Insights |
|--------|-----------|-----------|---------------|
| **Naïve Bayes** | **0.89** | **0.86** | Best trade-off between performance and computation time |
| **KNN** | **0.83** | **0.80** | Slightly lower accuracy, sensitive to distance metric |

### Key Findings:
- Frequent positive words include *amazing, love, great, excellent, beautiful*.  
- Frequent negative words include *boring, waste, bad, disappointing*.  
- Reviews show clear word-frequency patterns that correlate with sentiment polarity.

---

## ▶️ Running the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Fatimashamilova/movie-reviews-sentiment-analysis.git
cd movie-reviews-sentiment-analysis
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the Notebook

```bash
jupyter notebook Movie_reviewsProject.ipynb
```

---

## 📚 Project Structure

```
movie-reviews-sentiment-analysis/
│
├── Movie_reviewsProject.ipynb         # Main analysis notebook
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── .gitignore                         # Ignore environment files and caches
└── requirements.txt                   # Python dependencies
```

---

## 👩‍💻 Author

**Fatima Shamilova**

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

* **Kaggle IMDB Dataset Contributors** for making the dataset publicly available.
* **Bay Atlantic University** – for providing the academic foundation in NLP and Machine Learning.
* **Open-source Python community** – for powerful tools like NLTK and scikit-learn that make text analysis possible.
