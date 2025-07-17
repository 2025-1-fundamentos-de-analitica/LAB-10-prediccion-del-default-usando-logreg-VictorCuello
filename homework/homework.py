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

# CORRECCIÓN: Usar StratifiedKFold, es mejor para clasificación
from sklearn.model_selection import GridSearchCV, StratifiedKFold 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,  # CORRECCIÓN: Importar la métrica correcta
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

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

def clean_data(df):
    df_cleaned = df.copy()
    df_cleaned.rename(columns={"default payment next month": "default"}, inplace=True)
    if "ID" in df_cleaned.columns:
        df_cleaned.drop(columns=["ID"], inplace=True)
    
 
    mask = (df_cleaned["EDUCATION"] != 0) & (df_cleaned["MARRIAGE"] != 0)
    df_cleaned = df_cleaned.loc[mask].copy()
    
    df_cleaned.loc[df_cleaned["EDUCATION"] > 4, "EDUCATION"] = 4
    return df_cleaned

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
train_df_cleaned = clean_data(train_df.copy())
test_df_cleaned = clean_data(test_df.copy())

# --- Paso 2: Dividir en x_train, y_train, x_test, y_test ---
target_column = "default"
x_train = train_df_cleaned.drop(columns=[target_column])
y_train = train_df_cleaned[target_column]
x_test = test_df_cleaned.drop(columns=[target_column])
y_test = test_df_cleaned[target_column]

# --- Paso 3: Crear el Pipeline de Clasificación ---


categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = [col for col in x_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        
        ("scaler", MinMaxScaler(), numerical_features),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("classifier", LogisticRegression(random_state=42)), 
    ]
)


# --- Paso 4: Optimización de Hiperparámetros ---


param_grid = {
    'classifier__C': [0.7, 0.8, 0.9],
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__max_iter': [1500],
    'feature_selection__k': [1, 2, 5, 10]
}


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_ 

# --- Paso 5: Guardar el Modelo ---
with gzip.open(MODEL_FILE, "wb") as f:
    
    pickle.dump(grid_search, f)


# --- Paso 6 y 7: Calcular y Guardar Métricas ---
y_train_pred = best_model.predict(x_train)
y_test_pred = best_model.predict(x_test)

def get_metrics_dict(y_true, y_pred, dataset_name):
  
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y_true, y_pred), 
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


metrics_list = [
    get_metrics_dict(y_train, y_train_pred, "train"),
    get_metrics_dict(y_test, y_test_pred, "test"),
    get_cm_dict(y_train, y_train_pred, "train"),
    get_cm_dict(y_test, y_test_pred, "test")
]

with open(METRICS_FILE, "w") as f:
    for item in metrics_list:
        json.dump(item, f)
        f.write("\n")

print("\n¡Proceso completado exitosamente!")