from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

df = pd.read_csv("python_dataset_1000_rows.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(train_df['code'])

def recommend_hint(user_code):
    user_vec = vectorizer.transform([user_code])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_index = similarities.argmax()
    return train_df.iloc[best_index]['hint']

app = Flask(__name__)

@app.route("/")
def home():
    return "Hint Recommendation API is running on Vercel!"

@app.route("/get_hint", methods=["POST"])
def get_hint():
    data = request.json
    if "code" not in data:
        return jsonify({"error": "Please provide code"}), 400

    hint = recommend_hint(data["code"])
    return jsonify({"hint": hint})
