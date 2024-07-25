# Use the nvidia/cuda image as the base image for CUDA support
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Create a working directory
FROM python:3.8
WORKDIR /app
COPY requirements.txt .
# Install Python and other dependencies
RUN apt-get update && apt-get install -y python3-pip python3-dev && pip3 install --no-cache-dir -r requirements.txt


# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code to the container
COPY . .

# Set the entry point for the container
ENTRYPOINT [ "bash", "run_mvtec.sh" ]