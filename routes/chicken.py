from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from model import ChickenDiseaseClassifier
from config import Config
import io

router = APIRouter()

# ── Cấu hình model ──────────────────────────────────────────────────────────
CLASS_NAMES = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]
MODEL_PATH  = "models/best_model.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_model():
    cfg = Config()  # 🔥 phải giống lúc train
    model = ChickenDiseaseClassifier(cfg)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()
    return model

try:
    _model = load_model()
except Exception as e:
    print(f"[chicken] Không thể tải model: {e}")
    _model = None

# ── Endpoint ─────────────────────────────────────────────────────────────────
@router.post("/api/chicken/predict")
async def predict_chicken(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model chưa được tải. Kiểm tra file best_model.pth.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể đọc ảnh.")

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

    pred_idx    = int(torch.argmax(torch.tensor(probs)))
    prediction  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx]
    probabilities = {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)}

    return JSONResponse({
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "probabilities": probabilities,
    })
