# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501
import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --- Configuración Inicial ---
INPUT_DIR = "files/input/"
OUTPUT_DIR = "files/output/"
MODELS_DIR = "files/models/"
TRAIN_FILE = os.path.join(INPUT_DIR, "train_data.csv.zip")
TEST_FILE = os.path.join(INPUT_DIR, "test_data.csv.zip")
MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl.gz")
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# --- Paso 1: Limpieza de los Datasets ---

print("Paso 1: Realizando limpieza de los datasets...")

def clean_data(df):
    """
    Función para limpiar el dataframe siguiendo la interpretación literal
    de las instrucciones del ejercicio.
    """
    df_cleaned = df.copy()
    
    # Renombrar la columna objetivo
    df_cleaned.rename(columns={"default payment next month": "default"}, inplace=True)
    
    # Remover la columna 'ID'
    if "ID" in df_cleaned.columns:
        df_cleaned.drop(columns=["ID"], inplace=True)
        
    # --- CAMBIO FINAL Y CORRECTO ---
    # INSTRUCCIÓN 3: "Elimine los registros con informacion no disponible."
    # El diccionario define EDUCATION=0 y MARRIAGE=0 como N/A.
    # Se filtran estas filas para que el dataset coincida con el del calificador.
    mask = (df_cleaned["EDUCATION"] != 0) & (df_cleaned["MARRIAGE"] != 0)
    df_cleaned = df_cleaned.loc[mask].copy()
    
    # INSTRUCCIÓN 4: "Para la columna EDUCATION, valores > 4 ... agrupe estos valores en la categoría 'others' (4)."
    # Esto significa mapear cualquier valor de EDUCATION como 5, 6, etc., a 4.
    df_cleaned.loc[df_cleaned["EDUCATION"] > 4, "EDUCATION"] = 4
    
    return df_cleaned

# Cargar y limpiar datos
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

train_df_cleaned = clean_data(train_df.copy())
test_df_cleaned = clean_data(test_df.copy())

print(f"Datos de entrenamiento limpios: {train_df_cleaned.shape}")
print(f"Datos de prueba limpios: {test_df_cleaned.shape}")


# --- Paso 2: Dividir en x_train, y_train, x_test, y_test ---

print("\nPaso 2: Dividiendo los datos en X e y...")

target_column = "default"
x_train = train_df_cleaned.drop(columns=[target_column])
y_train = train_df_cleaned[target_column]
x_test = test_df_cleaned.drop(columns=[target_column])
y_test = test_df_cleaned[target_column]

print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")


# --- Paso 3: Crear el Pipeline de Clasificación ---

print("\nPaso 3: Creando el pipeline del modelo...")

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]
numerical_features = [col for col in x_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("scaler", MinMaxScaler(), numerical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')),
    ]
)

print("Pipeline creado exitosamente.")


# --- Paso 4: Optimización de Hiperparámetros ---

print("\nPaso 4: Optimizando hiperparámetros con GridSearchCV...")

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "feature_selection__k": [10, 15, 20, "all"],
}

cv = KFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(x_train, y_train)

print(f"\nMejores parámetros encontrados: {grid_search.best_params_}")
print(f"Mejor puntuación (balanced_accuracy) en CV: {grid_search.best_score_:.4f}")


# --- Paso 5: Guardar el Modelo ---

print("\nPaso 5: Guardando el modelo optimizado...")

with gzip.open(MODEL_FILE, "wb") as f:
    pickle.dump(grid_search, f)

print(f"Modelo guardado en: {MODEL_FILE}")


# --- Paso 6 y 7: Calcular y Guardar Métricas y Matrices de Confusión ---

print("\nPaso 6 y 7: Calculando y guardando métricas y matrices de confusión...")

y_train_pred = grid_search.predict(x_train)
y_test_pred = grid_search.predict(x_test)

def get_metrics_dict(y_true, y_pred, dataset_name):
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

def get_cm_dict(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }

train_metrics = get_metrics_dict(y_train, y_train_pred, "train")
test_metrics = get_metrics_dict(y_test, y_test_pred, "test")
train_cm = get_cm_dict(y_train, y_train_pred, "train")
test_cm = get_cm_dict(y_test, y_test_pred, "test")

with open(METRICS_FILE, "w") as f:
    json.dump(train_metrics, f)
    f.write("\n")
    json.dump(test_metrics, f)
    f.write("\n")
    json.dump(train_cm, f)
    f.write("\n")
    json.dump(test_cm, f)
    f.write("\n")

print(f"Métricas y matrices de confusión guardadas en: {METRICS_FILE}")
print("\n¡Proceso completado exitosamente!")