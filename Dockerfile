# Start with a slim Python base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the application
RUN useradd --create-home appuser
WORKDIR /home/appuser/app
USER appuser

# Copy and install Python requirements
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- CRUCIAL FIX ---
# Download the required spaCy model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Expose the port Streamlit runs on
EXPOSE 7860

# Set the command to run the application
# We use python -m streamlit instead of just streamlit to ensure it uses the user-installed package
CMD ["python", "-m", "streamlit", "run", "smartresearch_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

