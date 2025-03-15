FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models
COPY src ./src

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.inference:app"]
