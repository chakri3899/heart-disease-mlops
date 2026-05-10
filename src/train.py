import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay

from src.data_preprocessing import load_data, preprocess

# Ensure MLflow directory exists
os.makedirs("mlruns", exist_ok=True)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("heart-disease")

# Load dataset
df = load_data("data/heart.csv")

# Preprocess
X, y, scaler = preprocess(df)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_score = 0

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        print(f"\n{name} Results:")
        print(f"Accuracy: {acc}")
        print(f"ROC-AUC: {roc_auc}")
        print(classification_report(y_test, preds))

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        print(f"CV Accuracy: {cv_scores.mean()}")

        # MLflow logging
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("cv_accuracy", cv_scores.mean())

        #ROC Curve Plot
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"ROC Curve - {name}")
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()

        mlflow.sklearn.log_model(model, name)

        # Save best model
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model

# Save best model
joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nBest model saved!")