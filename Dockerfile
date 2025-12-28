# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
# We will copy the model at build time or runtime. 
# For CI/CD, we usually build the image AFTER training and include the model.
# Or we mount a volume. For this assignment's "container build" requirement, 
# we'll assume we duplicate the 'models' directory into the container.
COPY models/ models/

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
