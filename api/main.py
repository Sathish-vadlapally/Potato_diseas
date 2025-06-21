from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
from keras.layers import TFSMLayer  # ✅ This import was missing
import tensorflow as tf
import os

app = FastAPI()

# ✅ Load the SavedModel using TFSMLayer (Keras 3 compatible)

MODEL_PATH = os.path.join("saved_models", "1.keras")

MODEL = tf.keras.models.load_model(MODEL_PATH)
# ✅ Class names must match the model output
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return {"message": "hello I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).resize((256, 256))  # Resize if needed
    return np.array(image)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL(img_batch)  # ✅ use __call__ instead of .predict()
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {
        "class": predicted_class,
        "confidence": confidence
    }
