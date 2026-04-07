from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np

router = APIRouter()

# ── Schema — đúng 8 features ──────────────────────────────────────────────────
class GradeInput(BaseModel):
    Medu:     int    # Trình độ học vấn mẹ (0-4)
    Fedu:     int    # Trình độ học vấn cha (0-4)
    failures: int    # Số lần trượt môn (0-4)
    higher:   int    # Muốn học cao hơn (1/0)
    Walc:     int    # Uống rượu cuối tuần (1-5)
    absences: int    # Số buổi vắng mặt (0-93)
    G1:       float  # Điểm kỳ 1 (0-20)
    G2:       float  # Điểm kỳ 2 (0-20)

# ── Load bundle ───────────────────────────────────────────────────────────────
MODEL_PATH = "models/all_student_models.joblib"

MODEL_KEY_MAP = {
    "Linear Regression": "Linear Regression",
    "SVM (SVR)":         "SVM (SVR)",
    "Random Forest":     "Random Forest",
    "KNN":               "KNN",
}

try:
    _bundle = joblib.load(MODEL_PATH)
    _models = _bundle["model"]
    _scaler = _bundle["scaler"]
    print(f"[grade] Loaded {len(_models)} models: {list(_models.keys())}")
except Exception as e:
    print(f"[grade] Không thể tải model: {e}")
    _bundle  = None
    _models  = None
    _scaler  = None

# ── Endpoint ──────────────────────────────────────────────────────────────────
@router.post("/api/grade/predict/{model_name}")
async def predict_grade(model_name: str, data: GradeInput):
    if _models is None:
        raise HTTPException(status_code=503, detail="Model chưa được tải. Kiểm tra file all_student_models.joblib.")

    key = MODEL_KEY_MAP.get(model_name)
    if key is None or key not in _models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' không hợp lệ.")

    model = _models[key]

    raw = np.array([[
        data.Medu, data.Fedu, data.failures, data.higher,
        data.Walc, data.absences, data.G1, data.G2,
    ]], dtype=float)

    scaled = _scaler.transform(raw)

    try:
        prediction = float(model.predict(scaled)[0])
        prediction = max(0.0, min(20.0, prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")

    return JSONResponse({
        "predicted_G3": round(prediction, 2),
        "model_name":   model_name,
    })