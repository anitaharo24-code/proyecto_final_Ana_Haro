"""
Script de entrenamiento para el modelo final y evaluación básica.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# IMPORTANTE: Se debe importar los algoritmos que quieran usar, por ejemplo:
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

def train_best_model(processed_train_data_path: str, model_save_path: str):
    """
    INSTRUCCIONES:
    1. Carga los datos de entrenamiento procesados (que ya pasaron por `build_features.py`).
    2. Separa las características (X) de la etiqueta a predecir (y = 'median_house_value').
    3. Instancia tu mejor modelo encontrado después de la fase de experimentación y "fine tuning"
       (Por ejemplo: RandomForestRegressor con los mejores hiperparámetros).
    4. Entrena el modelo haciendo fit(X, y).
    5. Guarda el modelo entrenado en `model_save_path` (ej. 'models/best_model.pkl') usando joblib.dump().
    """
    #1. Cargar los datos de entrenamiento procesados
    df_train = pd.read_csv(processed_train_data_path)

    # 2. Separar las características (X) de la etiqueta a predecir (y)
    X_train = df_train.drop(columns=["median_house_value"])
    y_train = df_train["median_house_value"]

    # Escalar las características
    scaler = StandardScaler()
    numerical_features = X_train.select_dtypes(include="number").columns
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
  
    # 3. Instanciar el mejor modelo encontrado
    mejor_modelo = RandomForestRegressor(n_estimators=200, min_samples_split=2, random_state=42)

    # 4. Entrenar el modelo
    mejor_modelo.fit(X_train, y_train)
    print("Modelo entrenado exitosamente")

    # 5. Guardar el modelo entrenado
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor_modelo, model_save_path)
    joblib.dump(scaler, model_save_path.replace(".pkl", "_scaler.pkl"))
    print(f"Modelo guardado en: {model_save_path}")
    print(f"Scaler guardado en: {model_save_path.replace('.pkl', '_scaler.pkl')}")

def evaluate_model(model_path: str, processed_test_data_path: str):
    """
    INSTRUCCIONES:
    1. Carga el modelo guardado con joblib.load().
    2. Carga los datos de prueba preprocesados.
    3. Genera predicciones (y_pred) sobre los datos de prueba usando predict().
    4. Compara y_pred con las etiquetas reales calculando el RMSE y repórtalo en la terminal.
    """
    # 1. Cargar el modelo guardado
    modelo_cargado = joblib.load(model_path)
    scaler = joblib.load(model_path.replace(".pkl", "_scaler.pkl"))

    # 2. Cargar los datos de prueba preprocesados
    df_test = pd.read_csv(processed_test_data_path)
    X_test = df_test.drop(columns=["median_house_value"])
    y_test = df_test["median_house_value"]

    # Escalar las características de prueba usando el mismo scaler que el entrenamiento
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # 3. Generar predicciones sobre los datos de prueba
    y_pred = modelo_cargado.predict(X_test)

    # 4. Calcular el RMSE y reportarlo
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE en test: ${rmse_test:,.0f}")

if __name__ == "__main__":
    PROCESSED_TRAIN_PATH = "data/processed/train_processed.csv"
    PROCESSED_TEST_PATH = "data/processed/test_processed.csv"
    MODEL_OUTPUT_PATH = "models/best_model.pkl"
    train_best_model(PROCESSED_TRAIN_PATH, MODEL_OUTPUT_PATH)
    evaluate_model(MODEL_OUTPUT_PATH, PROCESSED_TEST_PATH)
    print("Script de entrenamiento finalizado.")