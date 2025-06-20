from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import io


app = Flask(__name__)

# Página principal
@app.route('/')
def home():
    return render_template('index.html')

# Vista de regresión lineal
@app.route('/regresion')
def regresion():
    return render_template('regresion.html')

# Vista de logistica
@app.route('/logistica')
def logistica():
    return render_template('logistica.html')

# Vista de Arbol
@app.route('/arbol')
def arbol():
    return render_template('arbol.html')

# Vista de K-Means Clustering
@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

# API para datos de regresión
@app.route('/api/regresion')
def api_regresion():
    df = pd.read_csv("Students Social Media Addiction.csv")
    X = df[['Avg_Daily_Usage_Hours']].values
    y = df['Mental_Health_Score'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    reales = [{"x": float(x[0]), "y": float(y)} for x, y in zip(X_test, y_test)]
    prediccion = [{"x": float(x[0]), "y": float(y)} for x, y in zip(X_test, y_pred)]

    return jsonify({"reales": reales, "prediccion": prediccion})

# API para regresión logística
@app.route('/api/logistica')
def api_logistica():
    df = pd.read_csv("Students Social Media Addiction.csv")
    X = df[['Avg_Daily_Usage_Hours']].values
    y = df['Affects_Academic_Performance'].apply(lambda x: 1 if x == "Yes" else 0).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LogisticRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X_test)

    reales = [{"x": float(x[0]), "y": int(y)} for x, y in zip(X_test, y_test)]
    predicciones = [{"x": float(x[0]), "y": int(y)} for x, y in zip(X_test, y_pred)]

    return jsonify({"reales": reales, "predicciones": predicciones})

@app.route('/arbol.png')
def arbol_png():
    df = pd.read_csv("Students Social Media Addiction.csv")
    X = df[['Avg_Daily_Usage_Hours']].values
    y = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0}).astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    modelo = DecisionTreeClassifier()
    modelo.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(25, 6))
    plot_tree(modelo, feature_names=['Avg_Daily_Usage_Hours'], class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=6, ax=ax)
    plt.title("Árbol de Decisión")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return send_file(buf, mimetype='image/png')

@app.route('/clustering.png')
def clustering_png():
    df = pd.read_csv("Students Social Media Addiction.csv")
    
    # Codificamos género
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])

    X = df[['Age', 'Gender', 'Addicted_Score']].values

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_

    # Gráfico 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        df['Age'], df['Gender'], df['Addicted_Score'],
        c=df['Cluster'], cmap='rainbow', s=70, edgecolor='black'
    )

    ax.set_title('Edad vs Género vs Puntaje de Adicción')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Género (0 = Female, 1 = Male)')
    ax.set_zlabel('Puntaje de Adicción')
    fig.colorbar(scatter, label='Grupo (Cluster)')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)