# 1. Use the exact version of Python we know works perfectly
FROM python:3.11.9-slim

# 2. Set the working directory inside the cloud server
WORKDIR /app

# 3. Install system-level Linux dependencies (This fixes the OCR!)
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev

# 4. Copy your requirements and install the Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your backend code into the server
COPY . .

# 6. Expose the port and start the FastAPI server
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
