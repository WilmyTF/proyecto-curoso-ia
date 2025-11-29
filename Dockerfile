# Usar una imagen base de Python ligera
FROM python:3.9-slim

# Directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY app.py .
COPY modelo_sentimientos.h5 .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 5000

# Comando para correr la app usando Gunicorn (servidor de producción)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]