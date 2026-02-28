Sentiment Analysis Tool 🧠📊

A machine learning–based sentiment analysis tool that classifies text as positive or negative using classical Natural Language Processing (NLP) techniques and supervised learning.

This project was developed as part of my internship tasks at SyntexHub.

📌 Project Overview

The tool performs the following steps:

Loads labeled text data (reviews / tweets)

Cleans and preprocesses text (tokenization, stopword removal)

Converts text into numerical features using TF-IDF

Trains a Logistic Regression classifier

Evaluates the model using Accuracy and F1-Score

Provides a Command Line Interface (CLI) for real-time sentiment prediction

🛠️ Technologies Used

Python

NLTK

Scikit-learn

Pandas

📂 Project Structure
sentiment-analysis/
├── data/
│   └── data_sample.csv
├── train.py
├── cli.py
├── requirements.txt
├── .gitignore
└── README.md
📊 Dataset

The full dataset is not included due to GitHub file size limits.

[A small sample dataset (data_sample.csv) is provided for demonstration and testing.](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Users can replace the sample file with the full dataset locally.

⚙️ Installation & Setup

Clone the repository:

git clone https://github.com/Youssef-Mohammad/Sentiment-Analysis-for-IMDB.git
cd sentiment-analysis

Install dependencies:

pip install -r requirements.txt

Download NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
🚀 How to Run
🔹 Train the Model
python train.py

This will:

Preprocess the data

Train the model

Evaluate performance

Save the trained model and vectorizer locally

🔹 Run the CLI
python main.py

Example:

Enter text: I really love this product
Predicted Sentiment: positive

Type exit to quit the CLI.

📈 Evaluation Metrics

Accuracy

F1-Score (weighted)

These metrics are printed after training to assess model performance.

📝 Notes

Trained model files (.pkl) and large datasets are excluded from version control.

This ensures the repository remains lightweight and reproducible.

Users can regenerate models by running train.py.

🌱 Future Improvements

Add neutral sentiment classification

Support more advanced models (Naive Bayes, SVM)

Build a GUI or web interface

Add confusion matrix and visualization

👤 Author

Youssef Mohammed
AI / Software Engineering Intern
