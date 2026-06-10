# Dockerfile — RL-Portfolio-Optimization
#
# Build:   docker build -t rl-portfolio .
# Run:     docker run --rm rl-portfolio
#          docker run --rm -p 5000:5000 rl-portfolio mlflow   ← MLflow UI

FROM python:3.10-slim

# System dependencies (needed by some torch/numpy builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir mlflow==2.13.0

# Copy the rest of the project
COPY . .

# Create directories that training scripts expect
RUN mkdir -p results data mlruns

# Default: run the full pipeline (quick test mode)
# Override CMD when running:
#   docker run rl-portfolio python mlflow_runner.py --dqn-episodes 500
CMD ["python", "mlflow_runner.py", \
     "--dqn-episodes", "50", \
     "--ppo-episodes", "20"]
