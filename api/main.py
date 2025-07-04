from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# ✅ Load the model saved in Keras format (.keras)

MODEL_PATH = "api/potato_disease_model.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)


# ✅ Class names (must match training order)
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

# ✅ Preprocessing function
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB").resize((224, 224))
    img_array = np.array(image).astype("float32") / 255.0  # Normalize
    return img_array

# ✅ Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)  # Shape: (1, 224, 224, 3)

    predictions = MODEL(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(float(np.max(predictions)), 2)

    return {
        "class": predicted_class,
        "confidence": confidence
    }
