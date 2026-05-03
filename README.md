# 🌱 Plant Disease Detection API (Stand Alone)

A FastAPI-based Machine Learning API that detects tomato plant diseases from leaf images using a trained TensorFlow model.

---

## 🚀 Overview

This API allows users to upload an image of a tomato leaf and receive:

- 🌿 Predicted disease name
- 📊 Confidence score
- 💊 Recommended remedy
- ⚠️ Warning if prediction confidence is low

The model was trained using a Convolutional Neural Network (CNN) and deployed using FastAPI.

---

## 🛠️ Technologies Used

- Python
- FastAPI
- TensorFlow / Keras
- NumPy
- Pillow
- Uvicorn

---

## 📦 Project Structure

```
plant-disease-api/
│
├── app.py
├── plant_disease_model.keras
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation (Local Setup)

1. Clone the repository:

```
git clone https://github.com/John31615/plant-disease-api.git
cd plant-disease-api
```

2. Create virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the API:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

5. Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 🌍 Deployment

This API can be deployed for free using:

- Render
- Railway
- Fly.io

Recommended start command:

```
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

## 📡 API Endpoint

### POST `/predict/`

Upload an image file of a tomato leaf.

### Example Response:

```json
{
  "disease": "Tomato_Early_blight",
  "confidence": 0.92,
  "remedy": "Remove infected leaves and apply fungicide.",
  "warning": "✅ Prediction confident."
}
```

---

## ⚠️ Notes

- The free hosting plan may sleep after inactivity.
- TensorFlow is memory-intensive, so performance may vary on free tiers.

---

## 📌 Future Improvements

- Convert model to TensorFlow Lite for lighter deployment
- Add authentication
- Connect to frontend/mobile application
- Improve model accuracy
- Add more plant classes

---

## 👨‍💻 Author

Developed as part of a Machine Learning project.

---

## 📜 License

This project is for educational purposes.
