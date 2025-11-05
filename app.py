from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
import joblib

app = FastAPI(title="School ML Service (Multi-tenant)")

# folder where your .pkl models live
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# in-memory cache so we don’t re-load from disk every time
MODEL_CACHE = {}

# adjust this list to match the features you used during training
EXPECTED_FEATURES = ["avg_score", "attendance_rate", "balance"]


class PredictRequest(BaseModel):
    school_id: str
    student_id: str
    features: Dict[str, Any]


def load_model_for_school(school_id: str):
    """
    1. Try to load a model for this school: models/student_performance_model_{school_id}.pkl
    2. If not found, use global model: models/student_performance_model_global.pkl
    3. Cache whatever we load.
    """

    # 1. cache hit
    if school_id in MODEL_CACHE:
        return MODEL_CACHE[school_id]

    # 2. try school-specific model
    school_model_path = os.path.join(MODELS_DIR, f"student_performance_model_{school_id}.pkl")
    if os.path.exists(school_model_path):
        model = joblib.load(school_model_path)
        MODEL_CACHE[school_id] = model
        return model

    # 3. try global model (cache under 'global')
    if "global" in MODEL_CACHE:
        return MODEL_CACHE["global"]

    global_model_path = os.path.join(MODELS_DIR, "student_performance_model_global.pkl")
    if os.path.exists(global_model_path):
        model = joblib.load(global_model_path)
        MODEL_CACHE["global"] = model
        return model

    # 4. nothing found
    raise FileNotFoundError("No model found for this school, and no global model available.")


def build_feature_vector(features: Dict[str, Any]):
    """
    Turn incoming dict into a list in the exact order the model expects.
    If something is missing, we fill with 0.
    """
    row = []
    for name in EXPECTED_FEATURES:
        val = features.get(name, 0)
        try:
            val = float(val)
        except Exception:
            val = 0.0
        row.append(val)
    return [row]  # sklearn wants 2D


@app.post("/predict/student")
def predict_student(req: PredictRequest):
    # try to load the right model
    try:
        model = load_model_for_school(req.school_id)
    except FileNotFoundError as e:
        # graceful fallback so your PHP doesn’t break
        return {
            "student_id": req.student_id,
            "school_id": req.school_id,
            "error": str(e),
            "predicted_score_next_term": req.features.get("avg_score", 0),
            "attendance_risk": "unknown",
            "source": "fallback-no-model"
        }

    # build feature vector
    X = build_feature_vector(req.features)

    # run prediction
    try:
        y_pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # simple rule-based attendance risk (optional)
    attendance_rate = float(req.features.get("attendance_rate", 1.0))
    if attendance_rate < 0.8:
        attendance_risk = "high"
    elif attendance_rate < 0.9:
        attendance_risk = "medium"
    else:
        attendance_risk = "low"

    # tell caller whether this was a school model or global
    school_model_filename = f"student_performance_model_{req.school_id}.pkl"
    model_files = os.listdir(MODELS_DIR) if os.path.isdir(MODELS_DIR) else []
    source = "school-specific" if school_model_filename in model_files else "global"

    return {
        "student_id": req.student_id,
        "school_id": req.school_id,
        "predicted_score_next_term": round(float(y_pred), 2),
        "attendance_risk": attendance_risk,
        "features_used": EXPECTED_FEATURES,
        "source": source
    }


@app.get("/health")
def health():
    return {"status": "ok"}
