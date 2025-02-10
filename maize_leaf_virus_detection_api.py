from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Load model once when starting server
try:
    model = YOLO('./maize_leaf_model - Copy.pt')
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def process_image(image_bytes: bytes) -> str:
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Run detection
        results = model(image_np)
        
        # Process results
        result_image = results[0].plot()
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return image_base64
    
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        processed_image = process_image(contents)
        
        return {
            "success": True,
            "image": processed_image
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)