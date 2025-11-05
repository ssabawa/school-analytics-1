from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

app = FastAPI(title="School ML Service")

class PredictRequest(BaseModel):
    school_id: str
    student_id: str
    features: Dict[str, Any]

@app.post("/predict/student")
def predict_student(req: PredictRequest):
    avg_score = float(req.features.get("avg_score", 50))
    attendance_rate = float(req.features.get("attendance_rate", 1.0))
    predicted_score = avg_score + attendance_rate * 5
    risk = "low"
    if attendance_rate < 0.8 or avg_score < 50:
        risk = "high"
    elif attendance_rate < 0.9:
        risk = "medium"
    return {
        "student_id": req.student_id,
        "predicted_score_next_term": round(predicted_score, 2),
        "attendance_risk": risk,
        "recommended_actions": [
            "Review attendance",
            "Share report with parent"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
