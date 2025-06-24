from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import io
from PIL import Image
from fastapi.responses import RedirectResponse

app = FastAPI ()

model = load_model('models/model.h5')
class_indices = {'RDA': 0, 'ZEMA': 1}
class_labels = {v: k for k, v in class_indices.items()}

class ImageInput(BaseModel):
    image_base64: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: str

def preprocess_base64_image(b64_str):
    try:
        img_data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img = img.resize((224, 224))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


@app.get ('/')
def index ():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=PredictionResponse)
def predict_image(input_data: ImageInput):
    img_array = preprocess_base64_image(input_data.image_base64)
    prediction = model.predict(img_array)

    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(prediction[0][predicted_class])

    return {
        "prediction": class_labels[predicted_class],
        "confidence": f"{confidence * 100:.2f}%",
    }