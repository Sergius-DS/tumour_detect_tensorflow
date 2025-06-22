# load_data_tensorflow.py

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def load_and_preprocess(base_path='images', categories=["Healthy", "Tumor"]):
    # Recolectar rutas y etiquetas
    image_paths = []
    labels = []
    for category in categories:
        category_path = os.path.join(base_path, category)
        if os.path.isdir(category_path):
            for image_name in os.listdir(category_path):
                image_paths.append(os.path.join(category_path, image_name))
                labels.append(category)
        else:
            print(f"Advertencia: No se encontró la carpeta '{category}' en '{category_path}'")
    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    print("DataFrame con rutas y etiquetas:")
    print(df.head())
    print("\nDistribución de clases:")
    print(df['label'].value_counts())

    # Codificación de etiquetas
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['label'])

    # División en entrenamiento y conjunto temporal
    X_train_orig, X_temp, y_train_orig, y_temp = train_test_split(
        df[['image_path']], df['category_encoded'], train_size=0.8, shuffle=True, random_state=42, stratify=df['category_encoded']
    )

    # División del conjunto temporal en validación y prueba
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42, stratify=y_temp
    )

    # Sobremuestreo en entrenamiento
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_orig, y_train_orig)

    # Crear DataFrames con las columnas necesarias
    train_df = pd.DataFrame({'image_path': X_train_resampled['image_path'], 'label': y_train_resampled})
    # Convertir 'label' a string para compatibilidad con flow_from_dataframe
    train_df['category_encoded'] = train_df['label'].astype(str)

    valid_df = pd.DataFrame({'image_path': X_valid['image_path'], 'label': y_valid})
    valid_df['category_encoded'] = valid_df['label'].astype(str)

    test_df = pd.DataFrame({'image_path': X_test['image_path'], 'label': y_test})
    test_df['category_encoded'] = test_df['label'].astype(str)

    return train_df, valid_df, test_df, label_encoder

# Opcional: carga desde pickle si quieres
def load_dataset(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_processed", action='store_true', help="Guardar datasets procesados")
    args = parser.parse_args()

    train_df, valid_df, test_df, label_encoder = load_and_preprocess()

    if args.save_processed:
        os.makedirs('data', exist_ok=True)
        train_df.to_pickle('data/train_df.pkl')
        valid_df.to_pickle('data/valid_df.pkl')
        test_df.to_pickle('data/test_df.pkl')
        with open('data/label_encoder.pkl', 'wb') as f:
            import pickle
            pickle.dump(label_encoder, f)
        print("Dataframes y label encoder guardados en 'data/'")