from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

app = Flask(__name__, template_folder="../frontend")
CORS(app)

DATA_FILE = "../dataset/spam.csv"

def train_model():
    df = pd.read_csv(DATA_FILE, encoding='utf-8', encoding_errors='ignore')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df = df[['label', 'message']]
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['message']
    y = df['label_num']

    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vect, y)

    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    return model, vectorizer

model, vectorizer = train_model()

@app.route("/")
def home():
    return render_template("index.html")  # Serve frontend directly

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    message = data.get("message", "")
    if message == "":
        return jsonify({"error": "No message provided"}), 400

    vect = vectorizer.transform([message])
    prediction = model.predict(vect)[0]
    label = "Spam" if prediction == 1 else "Ham"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(debug=True)
