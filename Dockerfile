# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the entire contents of the 'app' folder into the container's working directory
COPY app/ .

# Expose the port on which your Flask app runs
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]
