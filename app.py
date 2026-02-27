from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Load your trained model
model = load_model("plant_disease_model.keras")  

 
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]


# Remedies
remedies = {
    "Tomato_Bacterial_spot": "Remove infected leaves and apply copper-based bactericide.",
    "Tomato_Early_blight": "Remove infected leaves and apply fungicide.",
    "Tomato_Late_blight": "Use copper fungicide and improve airflow.",
    "Tomato_Leaf_Mold": "Ensure good air circulation and use fungicide.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicide regularly.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use insecticidal soap or neem oil.",
    "Tomato__Target_Spot": "Remove infected leaves and apply fungicide.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Remove infected plants and control whiteflies.",
    "Tomato__Tomato_mosaic_virus": "Use virus-free seeds and sanitize tools.",
    "Tomato_healthy": "No treatment needed."
}


# Initialize FastAPI
app = FastAPI(title="Plant Disease Detection API")

@app.get("/")
async def root():
    return {"message": "Plant Disease Detection API is running! Add /docs/ at the end of URL then USE /Predict/ to detect diseases."}

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array)
    pred_index = np.argmax(preds)
    disease = class_names[pred_index]
    confidence = float(np.max(preds))
    remedy = remedies.get(disease, "No remedy available.")
    
    warning = ""
    if confidence < 0.7:
        warning = "⚠️ Low confidence. Consider re-checking the leaf or consult an expert."
    else:
       warning = "✅ Prediction confident."

    return JSONResponse(content={
        "disease": disease,
        "confidence": confidence,
        "remedy": remedy,
        "warning": warning
    })
