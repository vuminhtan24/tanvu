from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import timm
from torchvision import transforms
from PIL import Image
import io
import time

router = APIRouter(prefix="/api/flood", tags=["flood"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# MODEL (IMPORTANT FIX)
# =========================
model = timm.create_model(
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    pretrained=False
)

state_dict = torch.load("models/flood.pth", map_location=device)
# Try non-strict load to tolerate minor classifier/head shape differences
load_result = model.load_state_dict(state_dict, strict=False)
# Log any missing or unexpected keys for debugging
if load_result.missing_keys or load_result.unexpected_keys:
    print("Swin model load warnings:", load_result)

model.to(device)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# =========================
# PREDICT
# =========================
@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    start = time.time()

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    elapsed = round(time.time() - start, 4)

    return JSONResponse({
        "label": "flood" if probs[1] > probs[0] else "no_flood",
        "confidence": {
            "flood": round(probs[1].item(), 4),
            "no_flood": round(probs[0].item(), 4),
        },
        "inference_time_s": elapsed,
        "device": str(device),
    })