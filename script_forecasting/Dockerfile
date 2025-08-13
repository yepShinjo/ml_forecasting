# Lightweight + build tools for Prophet (CmdStan)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Makassar

# System deps for Prophet/cmdstan build & pandas wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++ make curl git ca-certificates \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional but recommended) Pre-install CmdStan so Prophet is fast on first run
# Comment out this block if your build env has no Internet or you prefer runtime install
RUN python - <<'PY'
import cmdstanpy
# installs to /root/.cmdstan/cmdstan-<ver>
cmdstanpy.install_cmdstan(progress=True)
PY

# Copy scripts
COPY sales_forecast.py forecast_batch_with_args.py item_forecast_with_args.py /app/

# Default command does nothing; override in Swarm service `command: [...]`
CMD ["python", "-c", "print('Forecasting image ready. Override command in service.')"]
