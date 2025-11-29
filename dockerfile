# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir helps keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the current directory contents into the container
COPY . .

# Expose port 8000 to the outside world
EXPOSE 8000

# Define environment variable for Python to run in unbuffered mode
# (This ensures logs are printed to the console immediately)
ENV PYTHONUNBUFFERED=1

# Run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]