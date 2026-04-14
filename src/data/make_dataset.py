"""
Script para descargar y extraer los datos originales del proyecto.
"""

import os
import urllib.request
import tarfile
from pathlib import Path

def fetch_housing_data(housing_url: str, housing_path: str):
    """
    INSTRUCCIONES:
    1. Asegúrate de que el directorio `housing_path` exista (usa os.makedirs o Path.mkdir).
    2. Usa urllib.request.urlretrieve para descargar el archivo .tgz desde `housing_url`.
    3. Usa tarfile.open para extraer el contenido en `housing_path`.
    
    URL de los datos: "https://github.com/ageron/data/raw/main/housing.tgz"
    Ruta de destino recomendada: "data/raw/"
    """
    # 1. Crear el directorio si no existe
    Path(housing_path).mkdir(parents=True, exist_ok=True)
    
    # 2. Descargar el archivo .tgz
    housing_dir = Path(housing_path)
    tgz_path = housing_dir / "housing.tgz"
    print(f"Descargando datos desde {housing_url} ...")
    urllib.request.urlretrieve(housing_url, tgz_path)
    print("Descarga completada.")

    
    # 3. Extraer el contenido
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_dir)
    print(f"Datos extraídos en '{housing_dir}'")

if __name__ == "__main__":
    URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    PATH = "data/raw/"
    fetch_housing_data(URL, PATH)