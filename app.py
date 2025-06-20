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
import base64
import io
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler



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


@app.route('/EDA')
def info_view():
    # Cargar el archivo CSV real
    df = pd.read_csv('Students Social Media Addiction.csv')

    # Captura la salida de df.info() en un string
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    return render_template('EDA.html', info=info_str)

@app.route('/head')
def show_data():
    # Cargar el CSV real
    df = pd.read_csv('Students Social Media Addiction.csv')

    # Obtener los primeros 10 registros como HTML (puedes cambiar el número si deseas)
    table_html = df.head(10).to_html(classes='table table-striped', index=False)

    return render_template('head.html', table=table_html)


@app.route('/grafica')
def mostrar_grafica():
    df = pd.read_csv('Students Social Media Addiction.csv')

    # Crear gráfico
    plt.figure(figsize=(8, 5))
    df['Addicted_Score'].hist(bins=15, edgecolor='black')
    plt.title('Distribución del Puntaje de Adicción')
    plt.xlabel('Puntaje de Adicción')
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', alpha=0.75)

    # Guardar la imagen en un buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template('grafica.html', img_data=img_base64)

@app.route('/boxplot')
def mostrar_boxplot():
    df = pd.read_csv('Students Social Media Addiction.csv')

    # Crear boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df['Addicted_Score'])
    plt.title('Distribución y Outliers del Puntaje de Adicción')
    plt.xlabel('Puntaje de Adicción')

    # Guardar en buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template('boxplot.html', img_data=img_base64)


@app.route('/relacion')
def grafica_relacion():
    df = pd.read_csv('Students Social Media Addiction.csv')

    # Crear el countplot
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Relationship_Status', data=df)
    plt.title('Distribución de Estado de Relación')
    plt.xlabel('Estado de Relación')
    plt.ylabel('Frecuencia')

    # Convertir la gráfica a imagen base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template('relacion.html', img_data=img_base64)


@app.route('/correlacion')
def matriz_correlaciones():
    df = pd.read_csv('Students Social Media Addiction.csv')

    # Calcular la matriz de correlación
    corr_mat = df.corr(numeric_only=True)

    # Crear el heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlaciones')

    # Convertir imagen a base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template('correlacion.html', img_data=img_base64)

@app.route('/normalizacion')
def normalizacion_view():
    df = pd.read_csv('Students Social Media Addiction.csv')

    hola_test = [
        'Age', 'Addicted_Score', 'Mental_Health_Score',
        'Conflicts_Over_Social_Media', 'Sleep_Hours_Per_Night',
        'Avg_Daily_Usage_Hours'
    ]

    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df[hola_test] = scaler.fit_transform(df_normalized[hola_test])

    head_table = df[hola_test].head().to_html(classes='table table-bordered', index=False)
    min_vals = df[hola_test].min().to_frame(name='Mínimo').T.to_html(classes='table table-sm table-success', index=False)
    max_vals = df[hola_test].max().to_frame(name='Máximo').T.to_html(classes='table table-sm table-danger', index=False)

    return render_template('normalizacion.html', head=head_table, minimo=min_vals, maximo=max_vals)
    

    

if __name__ == '__main__':
    app.run(debug=True)
