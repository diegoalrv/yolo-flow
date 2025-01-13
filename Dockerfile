
FROM python:3.8-slim

# Instalar dependencias del sistema necesarias para OpenCV y Git
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

#RUN pip install --no-cache-dir -r requirements.txt

# Asegurar permisos para las carpetas de datos y resultados
RUN mkdir -p /app/data /app/output && chmod -R 777 /app/data /app/output

# Copiar el resto de los archivos del proyecto
COPY . .

# Comando por defecto para ejecutar el modelo o pruebas
CMD ["python", "/app/main.py"]
