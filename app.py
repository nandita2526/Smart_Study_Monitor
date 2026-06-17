from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle, json, pandas as pd, numpy as np

app = Flask(__name__)
CORS(app)

# Load model
with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

clf = bundle["clf"]
reg = bundle["reg"]
features = bundle["features"]
encoders = bundle["encoders"]

# Load stats
with open("stats.json") as f:
    stats = json.load(f)

# ---------------- ROUTES ---------------- #

@app.route('/')
def home():
    return "ML Project is running 🚀"

@app.route("/api/stats")
def get_stats():
    return jsonify(stats)

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    row = {}

    # Encode categorical
    for col, enc in encoders.items():
        val = data.get(col, enc.classes_[0])
        if val in enc.classes_:
            row[col + "_enc"] = int(enc.transform([val])[0])
        else:
            row[col + "_enc"] = 0

    # Numeric features
    numeric = ["age","Medu","Fedu","traveltime","studytime","failures",
               "famrel","freetime","goout","Dalc","Walc","health","absences"]

    for col in numeric:
        row[col] = float(data.get(col, 0))

    # Grades
    g1 = float(data.get("G1", 10))
    g2 = float(data.get("G2", 10))
    row["G1"] = g1
    row["G2"] = g2
    row["avg_grade"] = (g1 + g2) / 2
    row["grade_trend"] = g2 - g1

    # Prediction
    X = pd.DataFrame([row])[features]
    at_risk_prob = float(clf.predict_proba(X)[0][1])
    predicted_grade = float(reg.predict(X)[0])

    return jsonify({
        "at_risk": at_risk_prob > 0.5,
        "risk_probability": round(at_risk_prob * 100, 1),
        "predicted_grade": round(max(0, min(20, predicted_grade)), 1),
        "recommendation": get_recommendation(row, at_risk_prob, predicted_grade)
    })

# ---------------- LOGIC ---------------- #

def get_recommendation(row, risk, grade):
    tips = []

    if row["studytime"] < 2:
        tips.append("Increase weekly study time to at least 5–10 hours")

    if row["absences"] > 10:
        tips.append("Reduce absences — attendance strongly correlates with grades")

    if row["failures"] > 0:
        tips.append("Seek tutoring support to address past subject failures")

    if row["Walc"] > 3:
        tips.append("Reduce weekend alcohol consumption for better focus")

    if row["grade_trend"] < -2:
        tips.append("Grade is declining — consider speaking with an academic advisor")

    if not tips:
        tips.append("Keep up the great work! Maintain current study habits")

    return tips

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True, port=5000)