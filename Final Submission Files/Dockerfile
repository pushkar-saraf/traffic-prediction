# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install any required system packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Copy the Model
COPY gru_model.keras /app/gru_model.keras

# Install any required packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install protobuf==3.20.*

# Copy the rest of the application code into the container
COPY phase2.py .

# Expose the port that the Flask app will run on
EXPOSE 5001

# Command to run the application
CMD ["gunicorn", "phase2:app", "--bind", "0.0.0.0:5001"]

