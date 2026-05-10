import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data_preprocessing import load_data, preprocess

# Load dataset
df = load_data("data/heart.csv")

# Preprocess data
X, y, scaler = preprocess(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow experiment
with mlflow.start_run():

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Log in MLflow
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")

    # Save model & scaler
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

print("\nTraining complete! Model saved in 'models/' folder.")