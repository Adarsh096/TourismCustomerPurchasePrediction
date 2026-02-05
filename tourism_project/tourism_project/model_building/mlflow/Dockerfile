FROM python:3.9-slim

# Install dependencies
RUN pip install mlflow

# Set work directory
WORKDIR /app

# Expose the default MLflow port
EXPOSE 7860

# Start MLflow server
# Note: Use 0.0.0.0 to ensure it's accessible outside the container
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "7860"]

# --- HEALTHCHECK SECTION ---
# This runs every 10 seconds. 
# It marks the container as "healthy" only if the server returns a 200 OK.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -f http://localhost:7860/version || exit 1
