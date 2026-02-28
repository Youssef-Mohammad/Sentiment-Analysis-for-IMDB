import pickle
import nltk
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(tokens)

# Load model & vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Sentiment Analysis CLI (type 'exit' to quit)\n")

while True:
    text = input("Enter text: ")
    if text.lower() == "exit":
        break

    clean_text = preprocess(text)
    vec = vectorizer.transform([clean_text])
    prediction = model.predict(vec)[0]

    print("Predicted Sentiment:", prediction)
    print("-" * 30)