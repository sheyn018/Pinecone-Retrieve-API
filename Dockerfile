# Build stage (installs dependencies)
FROM python:3.8-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Runtime stage (slim image with app)
FROM python:3.8-slim
WORKDIR /app
COPY --from=builder /app/ .  # Copy only installed dependencies and app
EXPOSE 5000
CMD ["python", "app.py"]