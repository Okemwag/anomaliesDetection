# Use the official lightweight Debian image
FROM debian:bullseye-slim

# Install Python and other necessary tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    build-essential \
    gcc \
    python3-dev \
    libatlas-base-dev \
    meson \
    ninja-build && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables to prevent Python from writing pyc files to disk
# and to ensure output is sent straight to the terminal without buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip
RUN pip3 install --upgrade pip

# Create and set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install the dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . /app/

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]
