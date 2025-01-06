# Utiliza una imagen base de Python
FROM python:3.8-slim

# Instala las dependencias necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo requirements.txt y lo instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el contenido del proyecto a /app
COPY . .

# Comando para ejecutar el script principal
CMD ["python", "main.py"]
