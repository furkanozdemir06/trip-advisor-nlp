# 🏨 TripAdvisor Hotel Review Sentiment Analysis

An end-to-end **Natural Language Processing (NLP)** and **sentiment analysis** project that classifies hotel reviews as **positive or negative** using machine learning and neural networks.

The project combines a **Jupyter Notebook for exploratory analysis, text preprocessing, and deep learning** with an **interactive Streamlit application** where users can enter a hotel review and receive a real-time sentiment prediction.

## 📌 Project Overview

The main goal of this project is to analyze TripAdvisor hotel reviews and build a binary sentiment classification system for the hospitality industry.

The project focuses on the most extreme ratings:

- **1-star reviews → Negative**
- **5-star reviews → Positive**

By processing customer review text and training neural-network-based classifiers, the project demonstrates how NLP can be used to automatically evaluate guest feedback, brand sentiment, and customer satisfaction.

## 📊 Dataset

The project uses the file:

```text
tripadvisor_hotel_reviews.csv
```

The dataset contains **20,491 hotel reviews** with two original columns:

| Column | Description |
|---|---|
| `Review` | Written hotel review text |
| `Rating` | Customer rating from 1 to 5 |

The notebook reports no missing values in either column.

> The CSV file must be placed in the project root directory before running the notebook or Streamlit application.

## 🧹 Text Preprocessing

The notebook applies several NLP preprocessing steps before modeling:

- Convert text to lowercase
- Remove URLs
- Remove punctuation
- Remove numbers and digits
- Remove newline and carriage-return characters
- Remove extra whitespace
- Tokenize review text
- Remove stop words
- Remove rare words
- Apply lemmatization
- Join processed tokens back into cleaned text

The Streamlit application uses a lighter cleaning function that converts text to lowercase, removes URLs, and keeps alphabetic characters for live predictions.

## 🔎 Exploratory Data Analysis

The Jupyter Notebook (`TripAdvisor.ipynb`) includes:

- Dataset inspection and shape analysis
- Data type and missing-value checks
- Rating distribution visualization
- Review text preprocessing
- Most common word analysis
- Overall review word cloud
- 1-star review word cloud
- 5-star review word cloud
- Positive vs. negative sentiment preparation
- Model training and evaluation
- Training/validation accuracy and loss curves
- Classification report
- Confusion matrix

## 🧠 NLP & Deep Learning Model

For the notebook model, only **1-star and 5-star reviews** are used for binary classification.

### Text Vectorization

The reviews are converted into numerical features using `CountVectorizer` with:

- English stop-word filtering
- Unigrams and bigrams (`ngram_range=(1, 2)`)
- Maximum of 5,000 features

### Neural Network Architecture

The notebook uses a TensorFlow/Keras Artificial Neural Network:

```text
Input Layer
    ↓
Dense Layer — 128 neurons, ReLU
    ↓
Dropout — 0.2
    ↓
Dense Layer — 64 neurons, ReLU
    ↓
Output Layer — 1 neuron, Sigmoid
```

The model uses:

- Binary cross-entropy loss
- Adam optimizer
- Early stopping
- 20% test split
- Validation split during training

## 📈 Notebook Model Performance

The notebook reports approximately:

| Metric | Result |
|---|---:|
| Test Accuracy | **98%** |
| Macro F1 Score | **0.96** |
| Negative-class F1 | **0.93** |
| Positive-class F1 | **0.99** |
| Test Reviews | **2,095** |
| Misclassified Reviews | **41** |

The confusion matrix reported in the notebook is:

```text
                 Predicted
               Neg      Pos
Actual Neg      270       19
Actual Pos       22     1784
```

This shows strong classification performance on both positive and negative hotel reviews.

## 🖥️ Streamlit Sentiment Dashboard

The project also includes an interactive Streamlit application (`tripadvisor.py`).

The dashboard trains a `scikit-learn` **MLPClassifier** on 1-star and 5-star reviews and provides three main sections.

### 1. Try It

Users can type an English hotel review and receive:

- Positive or negative sentiment prediction
- Prediction confidence percentage
- Positive sentiment probability

Example:

```text
The room was clean and the staff were very friendly.
```

The app processes the text and returns a live sentiment prediction.

### 2. Data Insights

The dashboard includes interactive Plotly visualizations for:

- Number of reviews per rating
- Review length by rating
- Most frequent words in 1-star reviews
- Most frequent words in 5-star reviews

### 3. Model Performance

The model section displays:

- Test accuracy
- Macro F1 score
- Interactive confusion matrix

The Streamlit model uses:

- `CountVectorizer`
- English stop words
- Unigrams and bigrams
- Maximum of 5,000 features
- `MLPClassifier` with hidden layers `(128, 64)`
- Early stopping
- 80/20 train-test split

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Scikit-learn
- TensorFlow / Keras
- NLTK
- WordCloud
- Jupyter Notebook

## 📁 Project Structure

```text
tripadvisor-sentiment-analysis/
│
├── TripAdvisor.ipynb
├── tripadvisor.py
├── tripadvisor_hotel_reviews.csv
└── README.md
```

## ⚙️ Installation

Clone or download the project, open a terminal in the project directory, and install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn plotly streamlit scikit-learn tensorflow nltk wordcloud jupyter
```

Depending on your local NLTK setup, you may also need to download the required tokenizer and language resources before running the full notebook preprocessing pipeline.

Make sure `tripadvisor_hotel_reviews.csv` is located in the same directory as `tripadvisor.py` and `TripAdvisor.ipynb`.

## 🚀 Running the Project

### Run the Streamlit Application

```bash
streamlit run tripadvisor.py
```

Streamlit will start a local web server and open the sentiment analysis dashboard in your browser.

### Run the Jupyter Notebook

```bash
jupyter notebook TripAdvisor.ipynb
```

Then execute the notebook cells to reproduce the EDA, preprocessing, model training, and evaluation steps.

## 🎯 Project Purpose

This project demonstrates practical skills in:

- Natural Language Processing
- Text cleaning and preprocessing
- Tokenization and lemmatization
- Bag-of-Words feature extraction
- N-gram modeling
- Exploratory Data Analysis
- Neural network classification
- Model evaluation
- Sentiment analysis
- Interactive machine learning application development
- Communicating model results visually
---

Built with Python, NLP, and neural networks to understand hotel guest sentiment from TripAdvisor reviews. 🏨💬🧠
