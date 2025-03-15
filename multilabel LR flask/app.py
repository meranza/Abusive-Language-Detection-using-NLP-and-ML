from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load(r"D:\Project\Abusive Language AI\multilabel LR flask\model\multi_label_logistic_regression.pkl")
vectorizer = joblib.load(r"D:\Project\Abusive Language AI\multilabel LR flask\model\tfidf_vectorizer.pkl")

# Define labels
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("comment")

        if not user_input:
            return "Error: No comment provided", 400

        # Convert input text to numerical features
        input_vector = vectorizer.transform([user_input])

        # Get probability predictions
        probabilities = model.predict_proba(input_vector)[0]  # Extract first row

        # Convert to percentages
        percentage_scores = {label: round(prob * 100, 2) for label, prob in zip(labels, probabilities)}

        # Find the highest probability label
        most_likely_label = max(percentage_scores, key=percentage_scores.get)

        return render_template(
            "result.html",
            comment=user_input,
            results=percentage_scores,
            most_likely_label=most_likely_label
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
