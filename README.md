# Potato Disease Detection API

This is a Machine Learning API built with FastAPI that predicts potato diseases from leaf images.

## 🧠 Model Info

- Model used: TensorFlow/Keras model (`1.keras`)
- Trained on: Potato leaf disease dataset
- Output: Returns the class of disease (Early Blight, Late Blight, Healthy)

## 📁 Project Structure

```
Potato_diseas/
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── __pycache__/
├── Potato_diseas/
│   └── saved_model/
│       ├── 1.keras
│       ├── 2.keras
│       └── 3.keras
├── README.md
```

## 🚀 How to Run Locally

1. Clone the repo and navigate into it:
    ```bash
    cd Potato_diseas
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # For Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r api/requirements.txt
    ```

4. Run the FastAPI server:
    ```bash
    uvicorn api.main:app --reload
    ```

5. Open your browser and test:
    - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 📦 API Endpoint

### `POST /predict`

- Accepts: Image file
- Returns: Predicted disease class

## 🙋‍♂️ Author

Created by Sathish Vadlapally. Machine Learning Engineer.

---

⭐️ If you like this project, feel free to use it in your portfolio!
