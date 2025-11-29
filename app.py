from flask import Flask, request, jsonify
from flask_cors import CORS  # <--- NUEVO: Importar CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import numpy as np

app = Flask(__name__)
CORS(app) # <--- NUEVO: Esto permite que cualquier página web consulte tu API

# Cargar modelo
model = load_model('modelo_sentimientos.h5')

# Configuración del vocabulario (igual que antes)
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  
word_index["<UNUSED>"] = 3

def decodificar_texto(text):
    tokens = text.lower().split()
    secuencia = []
    for word in tokens:
        secuencia.append(word_index.get(word, 2))
    return secuencia

@app.route('/', methods=['GET'])
def home():
    return "API de Sentimientos Activa."

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.get_json()
    texto_usuario = datos.get('texto', '')
    
    if not texto_usuario:
        return jsonify({'error': 'No se envió texto'}), 400

    secuencia = decodificar_texto(texto_usuario)
    vector_entrada = pad_sequences([secuencia], maxlen=100)
    
    prediccion = model.predict(vector_entrada)
    valor = prediccion[0][0]
    
    sentimiento = "Positivo" if valor > 0.5 else "Negativo"
    
    return jsonify({
        'texto_recibido': texto_usuario,
        'sentimiento': sentimiento,
        'confianza': float(valor)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)