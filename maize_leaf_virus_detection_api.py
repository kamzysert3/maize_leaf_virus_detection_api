# -*- coding: utf-8 -*-
"""Maize-Leaf-Virus-Detection-API.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OmBy3bQvJOwaNYvtdMshKMFgaU2F36UJ
"""

# !pip install fastapi uvicorn opencv-python-headless ultralytics python-multipart pyngrok

from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64
# import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware

# Load YOLO model
model = YOLO('./maize_leaf_model.pt')

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image(file) -> np.ndarray:
    image = np.array(Image.open(BytesIO(file)).convert("RGB"))
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = read_image(image_data)

    # Run YOLO model inference
    results = model(image)

    # Convert processed image to base64 to return to frontend
    _, buffer = cv2.imencode('.jpg', results[0].plot())
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    return {"image": jpg_as_text}

# nest_asyncio.apply()

# from pyngrok import ngrok
# !ngrok config add-authtoken 2lF5wXiCnwiL7Gw02aZYf670KtN_6cgH5KX53byFqAiJbH7e6
# public_url = ngrok.connect(8000)
# print(f"Public URL: {public_url}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app)