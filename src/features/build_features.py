"""
Módulo para limpieza y enriquecimiento (Feature Engineering) usando funciones simples.
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Maneja los valores faltantes.
       Puedes llenarlos con la mediana de la columna.
    2. Retorna el DataFrame limpio.
    """
    df= df.copy()
    imputer = IterativeImputer(max_iter=10, random_state=42)
    columnas_numericas = df.select_dtypes(include="number").columns
    df[columnas_numericas] = imputer.fit_transform(df[columnas_numericas])
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    INSTRUCCIONES:
    1. Agrega nuevas variables derivando las existentes, por ejemplo:
       - 'rooms_per_household' = total_rooms / households
       - 'population_per_household' = population / households
       - 'bedrooms_per_room' = total_bedrooms / total_rooms
    2. Retorna el DataFrame enriquecido.
    """
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["population_per_household"] = df["population"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función orquestadora que toma el DataFrame crudo y aplica limpieza y enriquecimiento.
    """
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)
    
    # IMPORTANTE: Aquí los alumnos deberían añadir codificación de variables categóricas
    # (ej. get_dummies para 'ocean_proximity') si no usan Pipelines de Scikit-Learn.
    encoder = OrdinalEncoder()
    df_featured["ocean_proximity"] = encoder.fit_transform(df_featured[["ocean_proximity"]])
    
    return df_featured

if __name__ == "__main__":
    print("Módulo de feature engineering... ")
    
    df_train = pd.read_csv("./data/interim/train_set.csv")
    print(f"Dataset cargado: {df_train.shape}")

    df_procesado = preprocess_pipeline(df_train)

    os.makedirs("./data/processed", exist_ok=True)
    df_procesado.to_csv("./data/processed/train_processed.csv", index=False)

    print("Pipeline completado")
    print(f"Shape: {df_procesado.shape}")
    print(df_procesado.head())
