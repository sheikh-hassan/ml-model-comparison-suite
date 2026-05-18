"""
ASSIGNMENT 2: ML Algorithms Comparison - Flask Application
Compare KNN, Decision Tree, and Naïve Bayes on CSV and MNIST data
"""

from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent


def load_optional_joblib(*relative_parts):
    artifact_path = BASE_DIR.joinpath(*relative_parts)
    if artifact_path.exists():
        return joblib.load(artifact_path)
    return None


print("Loading models...")

csv_knn = load_optional_joblib("models", "csv", "knn_model.pkl")
csv_dt = load_optional_joblib("models", "csv", "decision_tree_model.pkl")
csv_nb = load_optional_joblib("models", "csv", "naive_bayes_model.pkl")
csv_scaler = load_optional_joblib("models", "csv", "scaler.pkl")

mnist_knn = load_optional_joblib("models", "mnist", "knn_model.pkl")
mnist_dt = load_optional_joblib("models", "mnist", "decision_tree_model.pkl")
mnist_nb = load_optional_joblib("models", "mnist", "naive_bayes_model.pkl")
mnist_scaler = load_optional_joblib("models", "mnist", "scaler.pkl")

stats = load_optional_joblib("models", "comparison_stats.pkl") or {
    "available": False,
    "message": "Training artifacts are not present in this workspace yet."
}

csv_models = {
    "KNN": csv_knn,
    "Decision Tree": csv_dt,
    "Naïve Bayes": csv_nb,
}

mnist_models = {
    "KNN": mnist_knn,
    "Decision Tree": mnist_dt,
    "Naïve Bayes": mnist_nb,
}

if all(model is not None for model in [csv_knn, csv_dt, csv_nb, csv_scaler, mnist_knn, mnist_dt, mnist_nb, mnist_scaler]):
    print("✓ All models loaded successfully!")
else:
    print("! Some model artifacts are missing; UI pages will still load, prediction endpoints will return 503 until training artifacts are restored.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=stats)


@app.route("/comparison")
def comparison():
    return render_template("comparison.html", stats=stats)


@app.route("/api/stats")
def get_stats():
    return jsonify(stats)


@app.route("/predict/csv/<algorithm>", methods=["POST"])
def predict_csv(algorithm):
    if csv_scaler is None or any(model is None for model in csv_models.values()):
        return jsonify({"success": False, "error": "CSV model artifacts are unavailable in this workspace."}), 503

    if algorithm not in csv_models:
        return jsonify({"success": False, "error": f"Algorithm {algorithm} not found"}), 400

    try:
        data = request.get_json() or {}
        features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        values = np.array([data[f] for f in features]).reshape(1, -1)
        values_scaled = csv_scaler.transform(values)

        model = csv_models[algorithm]
        prediction = model.predict(values_scaled)[0]

        try:
            proba = model.predict_proba(values_scaled)[0]
            probability = {
                "not_survived": round(float(proba[0]), 4),
                "survived": round(float(proba[1]), 4),
            }
            confidence = round(float(max(proba)), 4)
        except Exception:
            probability = None
            confidence = None

        return jsonify({
            "success": True,
            "algorithm": algorithm,
            "prediction": int(prediction),
            "prediction_label": "Survived" if prediction == 1 else "Did Not Survive",
            "probability": probability,
            "confidence": confidence,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/predict/mnist/<algorithm>", methods=["POST"])
def predict_mnist(algorithm):
    if mnist_scaler is None or any(model is None for model in mnist_models.values()):
        return jsonify({"success": False, "error": "MNIST model artifacts are unavailable in this workspace."}), 503

    if algorithm not in mnist_models:
        return jsonify({"success": False, "error": f"Algorithm {algorithm} not found"}), 400

    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400

        file = request.files["image"]
        img = Image.open(file.stream).convert("L")
        img_array = np.array(img).flatten().reshape(1, -1)
        img_scaled = mnist_scaler.transform(img_array)

        model = mnist_models[algorithm]
        prediction = model.predict(img_scaled)[0]

        try:
            proba = model.predict_proba(img_scaled)[0]
            probabilities = {str(i): round(float(prob), 4) for i, prob in enumerate(proba)}
            confidence = round(float(max(proba)), 4)
        except Exception:
            probabilities = None
            confidence = None

        return jsonify({
            "success": True,
            "algorithm": algorithm,
            "prediction": int(prediction),
            "probabilities": probabilities,
            "confidence": confidence,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/predict/all/csv", methods=["POST"])
def predict_all_csv():
    if csv_scaler is None or any(model is None for model in csv_models.values()):
        return jsonify({"success": False, "error": "CSV model artifacts are unavailable in this workspace."}), 503

    try:
        data = request.get_json() or {}
        features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        values = np.array([data[f] for f in features]).reshape(1, -1)
        values_scaled = csv_scaler.transform(values)

        results = {}
        for algo_name, model in csv_models.items():
            prediction = model.predict(values_scaled)[0]
            try:
                proba = model.predict_proba(values_scaled)[0]
                probability = {
                    "not_survived": round(float(proba[0]), 4),
                    "survived": round(float(proba[1]), 4),
                }
                confidence = round(float(max(proba)), 4)
            except Exception:
                probability = None
                confidence = None
            results[algo_name] = {
                "prediction": int(prediction),
                "probability": probability,
                "confidence": confidence,
            }

        return jsonify({"success": True, "predictions": results})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/predict/all/mnist", methods=["POST"])
def predict_all_mnist():
    if mnist_scaler is None or any(model is None for model in mnist_models.values()):
        return jsonify({"success": False, "error": "MNIST model artifacts are unavailable in this workspace."}), 503

    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image provided"}), 400

        file = request.files["image"]
        img = Image.open(file.stream).convert("L")
        img_array = np.array(img).flatten().reshape(1, -1)
        img_scaled = mnist_scaler.transform(img_array)

        results = {}
        for algo_name, model in mnist_models.items():
            prediction = model.predict(img_scaled)[0]
            try:
                proba = model.predict_proba(img_scaled)[0]
                probabilities = {str(i): round(float(prob), 4) for i, prob in enumerate(proba)}
                confidence = round(float(max(proba)), 4)
            except Exception:
                probabilities = None
                confidence = None
            results[algo_name] = {
                "prediction": int(prediction),
                "probabilities": probabilities,
                "confidence": confidence,
            }

        return jsonify({"success": True, "predictions": results})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/algorithms")
def get_algorithms():
    return jsonify({
        "algorithms": list(csv_models.keys()),
        "csv_algorithms": list(csv_models.keys()),
        "mnist_algorithms": list(mnist_models.keys()),
    })


@app.route("/test")
def test():
    return jsonify({
        "status": "ok",
        "models_loaded": {
            "csv_knn": csv_knn is not None,
            "csv_dt": csv_dt is not None,
            "csv_nb": csv_nb is not None,
            "mnist_knn": mnist_knn is not None,
            "mnist_dt": mnist_dt is not None,
            "mnist_nb": mnist_nb is not None,
        }
    })


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ASSIGNMENT 2: ML Algorithms Comparison - Flask Server")
    print("=" * 70)
    print("Starting server on http://localhost:5002")
    print("Dashboard: http://localhost:5002/dashboard")
    print("Comparison: http://localhost:5002/comparison")
    print("API Test: http://localhost:5002/test")
    print("=" * 70 + "\n")

    app.run(debug=True, port=5002)
