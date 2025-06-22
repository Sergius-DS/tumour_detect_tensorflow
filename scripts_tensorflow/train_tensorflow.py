# train_tensorflow.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import pickle

# Cargar DataFrames procesados
train_df = pd.read_pickle('data/train_df.pkl')
valid_df = pd.read_pickle('data/valid_df.pkl')
test_df = pd.read_pickle('data/test_df.pkl')

# Cargar label_encoder
with open('data/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Parámetros
batch_size = 32
img_size = (224, 224)

# Generadores de datos
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='image_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    shuffle=True
)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    x_col='image_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='image_path',
    y_col='category_encoded',
    target_size=img_size,
    class_mode='binary',
    color_mode='rgb',
    batch_size=batch_size,
    shuffle=False
)

# Importar la función build_model
from model import build_model

# Construir el modelo
model = build_model()

# Función para entrenar un ciclo completo (inicial + fine-tuning)
def train_model():
    # Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    ]

    initial_epochs = 10

    # Entrenamiento inicial
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=initial_epochs,
        callbacks=callbacks
    )

    # Fine-tuning: desbloquear capas si es necesario
    # Ejemplo: si tienes una base preentrenada
    # base_model = model.get_layer('nombre_de_la_base')
    # base_model.trainable = True

    # Recompilar con menor LR para fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    # Entrenamiento de fine-tuning
    history_fine = model.fit(
        train_generator,
        validation_data=valid_generator,
        initial_epoch=initial_epochs,
        epochs=total_epochs,
        callbacks=callbacks
    )

    # Guardar el modelo final
    model.save('models/final_model.h5')

    # Guardar historial completo
    training_history = {
        'initial_epochs': initial_epochs,
        'fine_tune_epochs': fine_tune_epochs,
        'total_epochs': total_epochs,
        'history': history.history,
        'history_fine': history_fine.history
    }

    with open('results/training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    train_model()

    print("Entrenamiento completo y modelos guardados.")