import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuración
VOCAB_SIZE = 10000
MAX_LEN = 100

print("Cargando datos...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print("Procesando secuencias...")
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

# Construir el modelo
model = Sequential([
    Embedding(VOCAB_SIZE, 16, input_length=MAX_LEN),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Entrenando...")
model.fit(x_train, y_train, epochs=3, validation_split=0.2)

# Guardar el modelo
model.save("modelo_sentimientos.h5")
print("¡Modelo guardado como 'modelo_sentimientos.h5'!")