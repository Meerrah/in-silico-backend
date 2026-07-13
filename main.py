import io
import cv2
import numpy as np
from PIL import Image
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# =====================================================================
# 1. INITIALIZATION & CONFIGURATION
# =====================================================================
app = FastAPI(
    title="In-Silico Backend Engine",
    description="Production API for OCR processing and health tracking",
    version="1.0.0"
)

# Enable CORS so your Vercel frontend can safely communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you can swap "*" for your specific Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 2. COMPUTER VISION PREPROCESSING PIPELINE
# =====================================================================
def clean_image_for_ocr(image_bytes: bytes) -> Image.Image:
    """
    Applies advanced statistical data cleaning methodologies to raw pixels.
    Resizes, removes background noise, and eliminates uneven lighting/shadows.
    """
    # Convert raw uploaded bytes into an OpenCV matrix (numpy array)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image file format.")
        
    # Step A: Convert image to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step B: Upscale the image (makes small text much easier for Tesseract to isolate)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    
    # Step C: Smooth out jagged edge artifacts with a light Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step D: Adaptive Thresholding 
    # Calculates localized pixel intensity blocks dynamically to flatten paper background 
    binary = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Step E: Convert the OpenCV binary matrix back into a PIL Image wrapper
    return Image.fromarray(binary)

# =====================================================================
# 3. API ENDPOINTS / ROUTES
# =====================================================================

@app.get("/")
def home():
    """Root route providing a helpful status check instead of a 404."""
    return {
        "status": "online",
        "message": "In-Silico Engine running successfully. Append /docs to the URL to view interactive endpoints."
    }

@app.post("/ledger/upload")
async def extract_medical_document(file: UploadFile = File(...)):
    """
    Accepts image file uploads, sanitizes them via the CV2 preprocessing filter,
    and runs highly structured OCR extraction.
    """
    # Prevent empty or wrong format files from hitting the pipeline
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        # Read the raw stream bytes from the frontend multipart form
        contents = await file.read()
        
        # Pass the bytes through the computer vision filtering steps
        cleaned_image = clean_image_for_ocr(contents)
        
        # Execute the layout extraction engine
        extracted_text = pytesseract.image_to_string(cleaned_image)
        
        # Clean up any chaotic white space text artifacts
        sanitized_text = extracted_text.strip()
        
        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": sanitized_text if sanitized_text else "No text could be structurally identified."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Engine Processing Failure: {str(e)}")

# Keep your existing auth and user data logic below this line if applicable:
# @app.post("/auth/login") ...
