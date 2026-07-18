import io
import re
import csv
import uuid
import sqlite3
import random
import time
import smtplib
from email.message import EmailMessage
from datetime import date
from typing import Optional, List
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# =====================================================================
# 1. INITIALIZATION & DATABASE STACK
# =====================================================================
app = FastAPI(
    title="In-Silico Backend Engine",
    description="Production API for OCR processing and health tracking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_db():
    conn = sqlite3.connect("clinical_data.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT, patient_id TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS profiles (patient_id TEXT PRIMARY KEY, blood_group TEXT, hemo_genotype TEXT, metabolic_profile TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS ledger (record_id TEXT PRIMARY KEY, patient_id TEXT, category TEXT, date_logged TEXT, primary_content TEXT, secondary_subtext TEXT, is_monetized INTEGER, flagged_for_review INTEGER, is_voided INTEGER)''')
    # NEW: Table for secure password resets
    cursor.execute('''CREATE TABLE IF NOT EXISTS otps (email TEXT PRIMARY KEY, code TEXT, expiry REAL)''')
    conn.commit()
    return conn

db = init_db()

# =====================================================================
# 2. COMPUTER VISION PREPROCESSING ENGINE (DYNAMIC)
# =====================================================================
def clean_image_for_ocr(image_bytes: bytes) -> Image.Image:
    """
    Dynamically preprocesses images.
    Keeps clean documents untouched while gently cleaning shadows from complex photos.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image file format.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate image variance to determine contrast/noise
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Resize (always beneficial for Tesseract)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # If the image is highly clear and clean, bypass heavy thresholding
    if variance < 100:
        return Image.fromarray(gray)

    # Apply a gentler bilateral filter
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Dialed-back Adaptive Thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    return Image.fromarray(binary)

# =====================================================================
# 3. AUTHENTICATION SCHEMAS & ENDPOINTS (SECURED)
# =====================================================================
class AuthPayload(BaseModel):
    email: str
    password: str

class ForgotPasswordPayload(BaseModel):
    email: str

class PasswordResetPayload(BaseModel):
    email: str
    otp: str
    new_password: str

def send_otp_email(receiver_email: str, otp_code: str):
    """
    Handles sending the confirmation code. Currently mocks the email by logging it
    to the server console so you can test it without setting up SMTP credentials yet.
    """
    print(f"\n{'='*50}")
    print(f"SECURITY ALERT: MOCK EMAIL SENT")
    print(f"To: {receiver_email}")
    print(f"Your In-Silico Password Reset Code is: {otp_code}")
    print(f"{'='*50}\n")

    # To make this live, configure a Gmail App Password and uncomment below:
    # try:
    #     msg = EmailMessage()
    #     msg.set_content(f"Your In-Silico Passport password reset code is: {otp_code}\nThis code expires in 15 minutes.")
    #     msg['Subject'] = "In-Silico Password Reset Code"
    #     msg['From'] = "YOUR_GMAIL_ADDRESS@gmail.com"
    #     msg['To'] = receiver_email
    #     with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    #         server.login("YOUR_GMAIL_ADDRESS@gmail.com", "YOUR_GMAIL_APP_PASSWORD")
    #         server.send_message(msg)
    # except Exception as e:
    #     print(f"SMTP Server Error: {e}")

@app.get("/")
def home():
    return {"status": "online", "message": "In-Silico Engine running successfully."}

@app.post("/auth/signup")
async def signup(payload: AuthPayload):
    cursor = db.cursor()
    cursor.execute("SELECT email FROM users WHERE email = ?", (payload.email,))
    if cursor.fetchone():
        raise HTTPException(status_code=400, detail="Email already registered")

    new_patient_id = f"PT-{str(uuid.uuid4().int)[:6]}"
    cursor.execute("INSERT INTO users (email, password, patient_id) VALUES (?, ?, ?)", (payload.email, payload.password, new_patient_id))
    cursor.execute("INSERT INTO profiles (patient_id, blood_group, hemo_genotype, metabolic_profile) VALUES (?, 'Unknown', 'Unknown', 'Unknown')", (new_patient_id,))
    db.commit()
    return {"message": "Account created", "patient_id": new_patient_id}

@app.post("/auth/login")
async def login(payload: AuthPayload):
    cursor = db.cursor()
    cursor.execute("SELECT password, patient_id FROM users WHERE email = ?", (payload.email,))
    user = cursor.fetchone()
    if not user or user[0] != payload.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful", "patient_id": user[1]}

@app.post("/auth/forgot-password")
async def request_password_reset(payload: ForgotPasswordPayload):
    cursor = db.cursor()
    cursor.execute("SELECT email FROM users WHERE email = ?", (payload.email,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="No account associated with this email.")

    otp = str(random.randint(100000, 999999))
    expiry = time.time() + 900  # Expires in 15 minutes

    cursor.execute("REPLACE INTO otps (email, code, expiry) VALUES (?, ?, ?)", (payload.email, otp, expiry))
    db.commit()

    send_otp_email(payload.email, otp)
    return {"message": "If that email exists, an OTP has been sent."}

@app.post("/auth/reset-password")
async def reset_password(payload: PasswordResetPayload):
    cursor = db.cursor()
    cursor.execute("SELECT code, expiry FROM otps WHERE email = ?", (payload.email,))
    record = cursor.fetchone()

    if not record:
        raise HTTPException(status_code=400, detail="No active password reset request found.")

    stored_otp, expiry = record
    if time.time() > expiry:
        raise HTTPException(status_code=400, detail="OTP has expired. Please request a new one.")
    if stored_otp != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP code.")

    cursor.execute("UPDATE users SET password = ? WHERE email = ?", (payload.new_password, payload.email))
    cursor.execute("DELETE FROM otps WHERE email = ?", (payload.email,))
    db.commit()
    return {"message": "Password updated successfully"}

# =====================================================================
# 4. PATIENT PROFILE SCHEMAS & ENDPOINTS
# =====================================================================
class PatientProfile(BaseModel):
    blood_group: str
    hemo_genotype: str
    metabolic_profile: str

@app.get("/profile/{patient_id}")
async def get_profile(patient_id: str):
    cursor = db.cursor()
    cursor.execute("SELECT blood_group, hemo_genotype, metabolic_profile FROM profiles WHERE patient_id = ?", (patient_id,))
    profile = cursor.fetchone()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"blood_group": profile[0], "hemo_genotype": profile[1], "metabolic_profile": profile[2]}

@app.post("/profile/{patient_id}")
async def update_profile(patient_id: str, profile: PatientProfile):
    cursor = db.cursor()
    cursor.execute("UPDATE profiles SET blood_group = ?, hemo_genotype = ?, metabolic_profile = ? WHERE patient_id = ?",
                  (profile.blood_group, profile.hemo_genotype, profile.metabolic_profile, patient_id))
    db.commit()
    return {"message": "Profile updated successfully"}

# =====================================================================
# 5. TOXICITY SCHEMAS & CALCULATIONS ENGINE
# =====================================================================
class DynamicTrialPayload(BaseModel):
    target_dosage_mg: float
    molecular_weight: float
    lipophilicity_logp: float

@app.post("/predict/dynamic/{patient_id}")
async def predict_dynamic_toxicity(patient_id: str, payload: DynamicTrialPayload):
    cursor = db.cursor()
    cursor.execute("SELECT metabolic_profile FROM profiles WHERE patient_id = ?", (patient_id,))
    profile_row = cursor.fetchone()
    genotype = profile_row[0] if profile_row else "Unknown"

    cursor.execute("SELECT secondary_subtext FROM ledger WHERE patient_id = ? AND is_voided = 0 ORDER BY rowid DESC", (patient_id,))
    ledger_rows = cursor.fetchall()

    patient_ast = 25.0
    patient_crp = 1.0

    for (subtext,) in ledger_rows:
        if not subtext: continue
        if patient_ast == 25.0:
            ast_match = re.search(r'(?:AST|Aminotransferase)[\s\w\(\)]*:\s*(\d+\.?\d*)', subtext)
            if ast_match: patient_ast = float(ast_match.group(1))
        if patient_crp == 1.0:
            crp_match = re.search(r'(?:CRP|Reactive Protein)[\s\w\(\)]*:\s*(\d+\.?\d*)', subtext)
            if crp_match: patient_crp = float(crp_match.group(1))

    confidence_score = 0.99
    if genotype == "Unknown":
        confidence_score = 0.60; genetic_modifier = 1.0; imputation_flag = True
    else:
        imputation_flag = False
        genetic_modifier = {"Ultra-Rapid": 0.4, "Extensive": 1.0, "Intermediate": 1.5, "Poor": 3.2}.get(genotype, 1.0)

    simulated_stress_index = (payload.target_dosage_mg * payload.lipophilicity_logp * (patient_ast / 25.0)) * genetic_modifier
    if simulated_stress_index > 400: safety_status = "CRITICAL RISK: Induced Hepatic Toxicity Expected"
    elif simulated_stress_index > 180: safety_status = "BORDERLINE: Elevated Hepatocellular Stress Detected"
    else: safety_status = "SAFE: Compound within calculated metabolic tolerance"

    return {
        "imputed_data": imputation_flag, "confidence_score": confidence_score,
        "biological_clearance_multiplier": genetic_modifier, "calculated_stress_index": round(simulated_stress_index, 2),
        "clinical_safety_status": safety_status, "dynamic_metrics_used": {"AST": patient_ast, "CRP": patient_crp}
    }

# =====================================================================
# 6. HEALTH DATA LEDGER INFRASTRUCTURE
# =====================================================================
class EHRRecord(BaseModel):
    record_id: str; patient_id: str; category: str; date_logged: date; primary_content: str
    secondary_subtext: Optional[str] = None; is_monetized: bool = False; flagged_for_review: bool = False; is_voided: bool = False

@app.post("/ledger/ingest")
async def ingest_health_record(record: EHRRecord):
    cursor = db.cursor()
    cursor.execute('''INSERT INTO ledger (record_id, patient_id, category, date_logged, primary_content, secondary_subtext, is_monetized, flagged_for_review, is_voided)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (record.record_id, record.patient_id, record.category, record.date_logged, record.primary_content, record.secondary_subtext, int(record.is_monetized), int(record.flagged_for_review), int(record.is_voided)))
    db.commit()
    return {"status": "Success"}

@app.get("/ledger/history/{patient_id}")
async def get_patient_history(patient_id: str):
    cursor = db.cursor()
    cursor.execute("SELECT * FROM ledger WHERE patient_id = ? ORDER BY rowid DESC", (patient_id,))
    rows = cursor.fetchall()

    records = []
    for r in rows:
        records.append({
            "record_id": r[0], "patient_id": r[1], "category": r[2], "date_logged": r[3],
            "primary_content": r[4], "secondary_subtext": r[5],
            "is_monetized": bool(r[6]), "flagged_for_review": bool(r[7]), "is_voided": bool(r[8])
        })
    return {"records": records}

@app.post("/ledger/void/{record_id}")
async def void_health_record(record_id: str):
    cursor = db.cursor()
    cursor.execute("UPDATE ledger SET is_voided = 1 WHERE record_id = ?", (record_id,))
    db.commit()
    return {"message": "Record permanently voided"}

# =====================================================================
# 7. PRODUCTION OCR ENGINE WITH COMPUTER VISION PIPELINE (OMNI-EXTRACTOR)
# =====================================================================
@app.post("/ledger/upload")
async def smart_upload_document(patient_id: str = Form(...), file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    file_bytes = await file.read()

    try:
        # Computer Vision Cleanup
        cleaned_image = clean_image_for_ocr(file_bytes)
        custom_config = r'--oem 3 --psm 6'
        raw_text = pytesseract.image_to_string(cleaned_image, config=custom_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Engine Processing Failure: {str(e)}")

    if not raw_text.strip():
        return {"status": "Failed", "detail": "Document is completely blank or unreadable."}

    # Extraction Pass 1: Standard Numeric Labs (e.g., "Glucose 92.0 mg/dL")
    pattern_numeric = r"([A-Za-z\s\,\(\)\-]+?)\s+(\d+\.\d+|\d+)\s*(?:[HL]\s*)?([a-zA-Z\/%]+)"
    # Extraction Pass 2: Qualitative/Categorical Labs (e.g., "Nitrites: Positive")
    pattern_categorical = r"([A-Za-z\s\,\(\)\-]+?):\s*([A-Za-z0-9\-\>]+(?:\s\([A-Za-z]\))?)"

    lines = raw_text.split('\n')
    extracted_tests = []

    for line in lines:
        clean_line = line.replace('|', '').replace('[', '').replace(']', '')
        
        # Try Numeric Match
        match_num = re.search(pattern_numeric, clean_line)
        if match_num:
            test_name = match_num.group(1).strip()
            value = match_num.group(2).strip()
            unit = match_num.group(3).strip()
            if len(test_name) > 3 and test_name.lower() not in ['patient name', 'parameter', 'result', 'unit', 'dob', 'gender', 'age']:
                extracted_tests.append(f"{test_name}: {value} {unit}")
            continue
            
        # Try Categorical Match
        match_cat = re.search(pattern_categorical, clean_line)
        if match_cat:
            test_name = match_cat.group(1).strip()
            value = match_cat.group(2).strip()
            if len(test_name) > 3 and test_name.lower() not in ['patient name', 'date collected', 'specimen type', 'patient id', 'age/gender', 'date']:
                extracted_tests.append(f"{test_name}: {value}")

    cursor = db.cursor()
    rec_id = f"tx-{str(uuid.uuid4().int)[:5]}"

    # Routing Path A: Structured Lab Data Found
    if extracted_tests:
        aggregated_results = " • ".join(extracted_tests)
        cursor.execute('''INSERT INTO ledger (record_id, patient_id, category, date_logged, primary_content, secondary_subtext, is_monetized, flagged_for_review, is_voided)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (rec_id, patient_id, "Lab Test", str(date.today()), "AI Extracted: Diagnostics Panel", aggregated_results, 0, 0, 0))
        db.commit()
        return {"status": "Success", "features_extracted": len(extracted_tests), "type": "Lab Data"}

    # Routing Path B: Free-Form Text Found (Treat as Doctor's Note)
    clean_text = " ".join([line.strip() for line in lines if len(line.strip()) > 8])
    if len(clean_text) < 20:
        return {"status": "Failed", "detail": "No readable metrics or clinical text could be identified."}

    summary = clean_text[:250] + "..." if len(clean_text) > 250 else clean_text
    
    cursor.execute('''INSERT INTO ledger (record_id, patient_id, category, date_logged, primary_content, secondary_subtext, is_monetized, flagged_for_review, is_voided)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (rec_id, patient_id, "Clinical Visit", str(date.today()), "AI Extracted: Clinical Document", summary, 0, 0, 0))
    db.commit()
    
    return {"status": "Success", "features_extracted": "Text Block", "type": "Clinical Note"}

# =====================================================================
# 8. METRICS EXPORT
# =====================================================================
@app.get("/ledger/export/{patient_id}")
async def export_patient_data(patient_id: str):
    cursor = db.cursor()
    cursor.execute("SELECT blood_group, hemo_genotype, metabolic_profile FROM profiles WHERE patient_id = ?", (patient_id,))
    profile = cursor.fetchone()

    cursor.execute("SELECT date_logged, category, primary_content, secondary_subtext FROM ledger WHERE patient_id = ? AND is_voided = 0 ORDER BY date_logged DESC", (patient_id,))
    ledger_records = cursor.fetchall()

    stream = io.StringIO()
    stream.write('\ufeff')
    writer = csv.writer(stream)

    writer.writerow(["IN-SILICO PASSPORT: CLINICAL DATA EXPORT"])
    writer.writerow(["Patient ID", patient_id])
    if profile:
        writer.writerow(["Blood Group", profile[0]])
        writer.writerow(["Genotype", profile[1]])
        writer.writerow(["Metabolic Profile", profile[2]])

    writer.writerow([])
    writer.writerow(["Date", "Category", "Event", "Clinical Notes/Values"])

    for record in ledger_records:
        writer.writerow([record[0], record[1], record[2], record[3]])

    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=health_passport_{patient_id}.csv"
    return response
