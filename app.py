from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI(title="Plant Disease Detection API")

# Global interpreter
interpreter = None
input_details = None
output_details = None


# Load TFLite model on startup
@app.on_event("startup")
def load_tflite_model():
    global interpreter, input_details, output_details

    interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite model loaded successfully!")


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


@app.get("/")
async def root():
    return {
        "message": "Plant Disease Detection API is running! Add /docs/ to test."
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get predictions
    preds = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(preds)
    disease = class_names[pred_index]
    confidence = float(np.max(preds))
    remedy = remedies.get(disease, "No remedy available.")

    if confidence < 0.7:
        warning = "⚠️ Low confidence. Consider re-checking the leaf."
    else:
        warning = "✅ Prediction confident."

    return JSONResponse(content={
        "disease": disease,
        "confidence": confidence,
        "remedy": remedy,
        "warning": warning
    })
