
import os
import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
from torchvision import transforms

from xray.ml.model.arch import Net


# --------------------------------------------------
# 1. APPLICATION SETUP
# --------------------------------------------------

app = FastAPI(title="LungCare AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "index.html")
MODEL_PATH = os.path.join(BASE_DIR, "xray_model.pth")

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

model = Net()

try:
    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("LungCare AI model loaded successfully!")

except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")


# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# Must match training image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 4. HOME PAGE
# --------------------------------------------------

@app.get("/")
async def home():

    if not os.path.exists(HTML_PATH):
        raise HTTPException(
            status_code=500,
            detail="index.html not found"
        )

    return FileResponse(HTML_PATH)


# --------------------------------------------------
# 5. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": ["NORMAL", "PNEUMONIA"]
    }


# --------------------------------------------------
# 6. PREDICTION API
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        # Open uploaded X-ray image
        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        print("\n========== IMAGE DETAILS ==========")
        print("Image size:", image.size)

        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Model prediction
        with torch.no_grad():

            raw_output = model(image_tensor)

            # Convert raw logits into probabilities
            probabilities = F.softmax(raw_output, dim=1)

            predicted_index = torch.argmax(
                probabilities,
                dim=1
            ).item()

            scores = probabilities[0].tolist()

        # Class mapping
        labels = {
            0: "Normal",
            1: "Pneumonia"
        }

        prediction = labels[predicted_index]

        print("Normal probability:", scores[0])
        print("Pneumonia probability:", scores[1])
        print("Predicted class:", prediction)
        print("===================================\n")

        return {
            "prediction": prediction,
            "class_index": predicted_index,
            "scores": {
                "normal": round(scores[0] * 100, 2),
                "pneumonia": round(scores[1] * 100, 2)
            },
            "message": (
                "Educational X-ray classification result. "
                "Not a medical diagnosis."
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        print("Prediction error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
