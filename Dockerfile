FROM python:3.11-slim

WORKDIR /app

# kopiranje kode in artefakte
COPY app.py /app/app.py
COPY artifacts /app/artifacts
COPY requirements.txt /app/requirements.txt

# dependencies
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5005

# Zaženemo Flask app
CMD ["python", "app.py"]
