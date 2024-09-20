import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from sklearn.decomposition import PCA
from io import BytesIO
from utils import get_image_list, load_image, apply_pca, image_to_base64
from typing import Optional

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...), n_components: int = Form(20)):
    # Load uploaded image
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Split image into blue, green, and red channels
    blue, green, red = cv2.split(image)

    # Perform PCA on blue channel to compute explained variance
    pca_temp = PCA().fit(blue)
    explained_variance_ratio = pca_temp.explained_variance_ratio_

    # Compute cumulative variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Check if requested PCA components exceed the image dimensions
    if n_components > blue.shape[0]:
        raise HTTPException(status_code=400, detail="Number of components exceeds image dimensions")

    # Perform PCA on the red, green, and blue channels
    redI = apply_pca(red, n_components)
    greenI = apply_pca(green, n_components)
    blueI = apply_pca(blue, n_components)

    # Reconstruct the image by merging the PCA-reduced channels
    re_image_bgr = (np.dstack((blueI, greenI, redI))).astype(np.uint8)

    # Prepare a response with the variance preserved and the compressed image as base64 strings
    response = {
        "variance_preserved": f"{cumulative_variance[n_components - 1]:.2%}",
        "original_image": image_to_base64(image),

        "compressed_image": image_to_base64(re_image_bgr)
    }

    return JSONResponse(content=response)

