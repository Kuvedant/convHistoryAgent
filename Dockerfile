# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirement file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY backend.py /app/
COPY frontend.py /app/
COPY text_1.json /app/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
