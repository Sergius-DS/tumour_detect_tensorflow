# evaluate_tensorflow.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import argparse
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Crear generador para evaluación
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

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

# Cargar modelo entrenado
model = tf.keras.models.load_model('models/final_model.h5')

# Función para evaluar
def evaluate_model(model, generator, label_encoder, output_dir):
    # Predicciones
    preds = model.predict(generator, verbose=1)
    y_probs = preds.squeeze()
    y_true = generator.labels  # etiquetas verdaderas

    y_pred = (y_probs > 0.5).astype(int)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Matriz de Confusión - TensorFlow')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Reporte de clasificación
    print("Reporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - TensorFlow')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

# Función para graficar historial
def plot_training_history(history, initial_epochs, output_path):
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    train_loss = history['loss']
    val_loss = history['val_loss']
    total_epochs = len(train_acc)

    plt.figure(figsize=(12, 6))
    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Exactitud de Entrenamiento')
    plt.plot(val_acc, label='Exactitud de Validación')
    plt.axvline(x=initial_epochs - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Pérdida de Entrenamiento')
    plt.plot(val_loss, label='Pérdida de Validación')
    plt.axvline(x=initial_epochs - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results", help="Carpeta donde guardar gráficos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluar el modelo
    evaluate_model(model, test_generator, label_encoder, args.output_dir)

    # Cargar y graficar historia de entrenamiento
    training_history_path = os.path.join('results', 'training_history.pkl')
    if os.path.exists(training_history_path):
        try:
            with open(training_history_path, 'rb') as f:
                training_history = pickle.load(f)
            # La estructura en tu train_tensorflow.py es: 
            # {'initial_epochs':..., 'fine_tune_epochs':..., 'total_epochs':..., 'history':..., 'history_fine':...}
            history_initial = training_history['history']
            history_fine = training_history['history_fine']
            # Puedes concatenar los historiales si quieres graficar toda la curva
            train_acc = history_initial['accuracy'] + history_fine['accuracy']
            val_acc = history_initial['val_accuracy'] + history_fine['val_accuracy']
            train_loss = history_initial['loss'] + history_fine['loss']
            val_loss = history_initial['val_loss'] + history_fine['val_loss']
            initial_epochs = training_history['initial_epochs']

            plot_training_history(
                {
                    'accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'loss': train_loss,
                    'val_loss': val_loss
                }, 
                initial_epochs, 
                os.path.join(args.output_dir, 'training_history_plot.png')
            )
            print("Gráfica de entrenamiento guardada.")
        except Exception as e:
            print(f"Error al cargar o graficar historia: {e}")
    else:
        print("No se encontró training_history.pkl. Ejecuta train_tensorflow.py primero.")