# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port for Flask
EXPOSE 8080

# Default command
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
