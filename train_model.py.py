import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import pickle
import json

df = pd.read_csv("student-mat.csv", sep=";")

# Feature engineering
df["avg_grade"] = (df["G1"] + df["G2"]) / 2
df["grade_trend"] = df["G2"] - df["G1"]
df["study_efficiency"] = df["G3"] / (df["studytime"] + 1)
df["at_risk"] = (df["G3"] < 10).astype(int)

# Encode categoricals
le = LabelEncoder()
cat_cols = ["school", "sex", "address", "famsize", "Pstatus",
            "Mjob", "Fjob", "reason", "guardian", "schoolsup",
            "famsup", "paid", "activities", "nursery", "higher",
            "internet", "romantic"]
encoders = {}
for col in cat_cols:
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

features = ["age", "Medu", "Fedu", "traveltime", "studytime",
            "failures", "famrel", "freetime", "goout", "Dalc",
            "Walc", "health", "absences", "avg_grade", "grade_trend",
            "G1", "G2"] + [c + "_enc" for c in cat_cols]

X = df[features]
y_class = df["at_risk"]
y_reg = df["G3"]

X_train, X_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Train classifier (at-risk detection)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, yc_train)
print("Classifier Report:\n", classification_report(yc_test, clf.predict(X_test)))

# Train regressor (final grade prediction)
reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, yr_train)
preds = reg.predict(X_test)
print(f"Grade Predictor RMSE: {np.sqrt(mean_squared_error(yr_test, preds)):.2f}")

# Feature importance
importance = dict(zip(features, clf.feature_importances_))
top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

# Save everything
with open("model.pkl", "wb") as f:
    pickle.dump({"clf": clf, "reg": reg, "features": features, "encoders": encoders}, f)

with open("feature_importance.json", "w") as f:
    json.dump(top_features, f)

# Save summary stats for dashboard
stats = {
    "total_students": len(df),
    "at_risk_count": int(df["at_risk"].sum()),
    "avg_final_grade": round(df["G3"].mean(), 2),
    "pass_rate": round((df["G3"] >= 10).mean() * 100, 1),
    "avg_study_time": round(df["studytime"].mean(), 2),
    "avg_absences": round(df["absences"].mean(), 1),
    "grade_distribution": df["G3"].value_counts().sort_index().to_dict(),
    "studytime_vs_grade": df.groupby("studytime")["G3"].mean().round(2).to_dict(),
    "failures_vs_grade": df.groupby("failures")["G3"].mean().round(2).to_dict(),
    "top_features": top_features,
    "grade_trend": {
        "g1_avg": round(df["G1"].mean(), 2),
        "g2_avg": round(df["G2"].mean(), 2),
        "g3_avg": round(df["G3"].mean(), 2),
    }
}

with open("stats.json", "w") as f:
    json.dump(stats, f)

print("✅ Model trained and saved! Run: python app.py")