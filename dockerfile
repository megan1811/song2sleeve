# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.10.4
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Copy only pyproject files first to leverage Docker layer caching
COPY pyproject.toml poetry.lock* /app/

# Install dependencies with Poetry (no dev deps)
RUN poetry install --no-interaction --no-ansi --without dev

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8501

# Run the application.
CMD ["poetry", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
