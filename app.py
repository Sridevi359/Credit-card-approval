from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    income = float(request.form["income"])
    employment = int(request.form["employment"])
    credit_score = int(request.form["credit_score"])
    existing_loan = int(request.form["existing_loan"])
    dependents = int(request.form["dependents"])

    data = np.array([[age, income, employment, credit_score, existing_loan, dependents]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "✅ Credit Card Approved"
        color = "green"
    else:
        result = "❌ Credit Card Rejected"
        color = "red"

    return render_template("index.html", prediction=result, color=color)

if __name__ == "__main__":
    app.run(debug=True)