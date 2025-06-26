from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree


app = Flask(__name__)

# Cargar datos globalmente
df = pd.read_csv("Students Social Media Addiction.csv")

# Modelo de Regresión Lineal (Ej: predecir Mental_Health_Score)
modelo_lineal = LinearRegression()
modelo_lineal.fit(df[['Avg_Daily_Usage_Hours']], df['Mental_Health_Score'])

# Modelo de Regresión Logística (Ej: predecir si afecta rendimiento)
modelo_logistico = LogisticRegression()
modelo_logistico.fit(df[['Avg_Daily_Usage_Hours']], df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0}))

@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/lineal', methods=['GET', 'POST'])
def linea_dashboard():
    df = pd.read_csv("Students Social Media Addiction.csv")
    modelo = LinearRegression()
    modelo.fit(df[['Avg_Daily_Usage_Hours']], df['Mental_Health_Score'])

    datos_reales = [{"x": float(x), "y": float(y)} for x, y in zip(df['Avg_Daily_Usage_Hours'], df['Mental_Health_Score'])]

    if request.method == 'POST':
        horas = float(request.form['horas'])
        pred = modelo.predict([[horas]])[0]
        punto_usuario = {"x": horas, "y": round(pred, 2)}
        return render_template("lineal_dashboard.html", datos_reales=datos_reales, punto_usuario=punto_usuario, resultado=round(pred, 2), horas=horas)

    # En GET muestra sin punto del usuario
    return render_template("lineal_dashboard.html", datos_reales=datos_reales, punto_usuario={"x": None, "y": None})

@app.route('/logistica', methods=['GET', 'POST'])
def logistica_dashboard():
    df = pd.read_csv("Students Social Media Addiction.csv")
    X = df[['Avg_Daily_Usage_Hours']].values
    y = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0}).values

    modelo = LogisticRegression()
    modelo.fit(X, y)

    datos_reales = [{"x": float(x[0]), "y": int(y_)} for x, y_ in zip(X, y)]

    # Curva logística
    x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_probs = modelo.predict_proba(x_vals)[:, 1]
    curva_logistica = [{"x": float(x[0]), "y": float(y)} for x, y in zip(x_vals, y_probs)]

    # Variables por defecto
    punto_usuario = {"x": None, "y": None}
    resultado = None
    horas = None

    if request.method == 'POST':
        horas = float(request.form['horas'])
        pred = int(modelo.predict([[horas]])[0])
        resultado = "Sí afecta el rendimiento" if pred == 1 else "No afecta el rendimiento"
        punto_usuario = {"x": float(horas), "y": int(pred)}

    return render_template("logistica_dashboard.html",
                           datos_reales=datos_reales,
                           curva_logistica=curva_logistica,
                           punto_usuario=punto_usuario,
                           resultado=resultado,
                           horas=horas)

@app.route('/arbol', methods=['GET', 'POST'])
def arbol_dashboard():
    df = pd.read_csv("Students Social Media Addiction.csv")

    # Variables que vamos a usar
    features = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']
    target = 'Affects_Academic_Performance'

    # Preparamos los datos
    X = df[features]
    y = df[target].map({'Yes': 1, 'No': 0}).astype(int)

    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X, y)

    # Por defecto, sin datos del usuario
    resultado = None
    datos_usuario = {}

    if request.method == 'POST':
        try:
            datos_usuario = {
                'Avg_Daily_Usage_Hours': float(request.form['horas']),
                'Sleep_Hours_Per_Night': float(request.form['sueño']),
                'Conflicts_Over_Social_Media': int(request.form['conflictos'])
            }
            entrada = pd.DataFrame([datos_usuario])
            pred = modelo.predict(entrada)[0]
            resultado = "Sí afecta el rendimiento" if pred == 1 else "No afecta el rendimiento"
        except Exception as e:
            resultado = f"Error en la predicción: {str(e)}"

    # Generamos imagen del árbol
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_tree(modelo, feature_names=features, class_names=["No", "Sí"], filled=True, rounded=True, fontsize=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return render_template("arbol_dashboard.html", resultado=resultado, datos_usuario=datos_usuario, arbol_img=img_base64)

@app.route('/clustering', methods=['GET', 'POST'])
def clustering_dashboard():
    df = pd.read_csv("Students Social Media Addiction.csv")

    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])

    X = df[['Age', 'Gender', 'Addicted_Score']].values
    modeloG = KMeans(n_clusters=3, n_init=10, random_state=42)
    modeloG.fit(X)
    df['Cluster'] = modeloG.labels_

    resultado = None
    datos_usuario = {}

    if request.method == 'POST':
        try:
            datos_usuario = {
                'Age': float(request.form['edad']),
                'Gender': int(request.form['genero']),
                'Addicted_Score': float(request.form['adiccion'])
            }
            entrada = np.array([[datos_usuario['Age'], datos_usuario['Age'], datos_usuario['Addicted_Score']]])
            cluster_usuario = int(modeloG.predict(entrada)[0])
            resultado = f"Perteneces al Grupo (Cluster): {cluster_usuario}"
        except Exception as e:
            resultado = f"Error en la predicción: {str(e)}"

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(df['Age'], df['Gender'], df['Addicted_Score'],
                     c=df['Cluster'], cmap='rainbow', s=70, edgecolors='black')

    if datos_usuario:
        ax.scatter(datos_usuario['Age'], datos_usuario['Gender'], datos_usuario['Addicted_Score'],
                   c='black', s=200, marker='*', label='Tu', edgecolors='white')
    
    ax.set_title('Edad vs Género vs Adicción')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Género (0=F, 1=M)')
    ax.set_zlabel('Adicción')

    fig.colorbar(scatter, label='Grupo (Cluster)')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    arbol_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return render_template("clustering_dashboard.html",
                           resultado=resultado,
                           datos_usuario=datos_usuario,
                           arbol_img=arbol_img)

if __name__ == '__main__':
    app.run(debug=True)
