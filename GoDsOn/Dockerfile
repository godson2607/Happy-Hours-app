# Use a lightweight official Python image as the base
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the GoDsOn directory which contains your application and requirements.txt
# This copies the entire 'GoDsOn' directory and its contents into '/app/GoDsOn' in the container
COPY hh /app/hh

# Change the working directory to where your requirements.txt and main3.py are located
WORKDIR /app/hh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your FastAPI application will run on
EXPOSE 8000

# Command to run the application using Uvicorn
# The 'main3:app' assumes main3.py is directly in the current working directory (/app/GoDsOn)
CMD ["uvicorn", "main3:app", "--host", "0.0.0.0", "--port", "8000"]