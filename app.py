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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
def formulario():
    return render_template('formulario.html')

@app.route('/dashboard')
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

    
    return render_template("lineal_dashboard.html", datos_reales=datos_reales, punto_usuario={"x": None, "y": None})

@app.route('/logistica', methods=['GET', 'POST'])
def logistica_dashboard():
    df = pd.read_csv("Students Social Media Addiction.csv")
    X = df[['Avg_Daily_Usage_Hours']].values
    y = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0}).values

    modelo = LogisticRegression()
    modelo.fit(X, y)

    datos_reales = [{"x": float(x[0]), "y": int(y_)} for x, y_ in zip(X, y)]

    
    x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_probs = modelo.predict_proba(x_vals)[:, 1]
    curva_logistica = [{"x": float(x[0]), "y": float(y)} for x, y in zip(x_vals, y_probs)]

    
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

    
    features = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']
    target = 'Affects_Academic_Performance'

    
    X = df[features]
    y = df[target].map({'Yes': 1, 'No': 0}).astype(int)

    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X, y)

    
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

@app.route('/plataforma', methods=['GET', 'POST'])
def plataforma_dashboard():
    df = pd.read_csv('Students Social Media Addiction.csv')

    le_platform = LabelEncoder()
    le_gender = LabelEncoder()
    df['Most_Used_Platform'] = le_platform.fit_transform(df['Most_Used_Platform'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    X = df[['Age', 'Gender', 'Avg_Daily_Usage_Hours', 'Addicted_Score']] 
    y = df['Most_Used_Platform']

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    resultado = None
    datos_usuario = {}
    grafica_img = None

    if request.method == 'POST':
        try:
            datos_usuario = {
                'Age': float(request.form['edad']),
                'Gender': int(request.form['genero']),
                'Avg_Daily_Usage_Hours': float(request.form['horas']),
                'Addicted_Score': float(request.form['adiccion'])
            }
            entrada = pd.DataFrame([datos_usuario])
            probs = modelo.predict_proba(entrada)[0]
            clases = le_platform.inverse_transform(modelo.classes_)
            pred = modelo.predict(entrada)[0]
            resultado = le_platform.inverse_transform([pred])[0]
        except Exception as e:
            resultado = f"Error en la predicción: {str(e)}"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(clases, probs, color="#6366f1")
        ax.set_ylim(0, 1)
        ax.set_title("Probabilidad por Plataforma Usada")
        ax.set_ylabel("Probabilidad")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelrotation=15)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        grafica_img = base64.b64encode(buf.read()).decode('utf-8')

    return render_template("plataforma_dashboard.html",
                           resultado=resultado,
                           datos_usuario=datos_usuario,
                           grafica_img=grafica_img)

@app.route('/relacion_estado', methods=['GET', 'POST'])
def relacion_estado_dashboard():
    df = pd.read_csv("Students Social Media Addiction.csv")

    le_status = LabelEncoder()
    le_gender = LabelEncoder()

    df['Relationship_Status'] = le_status.fit_transform(df['Relationship_Status'])
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    features = ['Gender', 'Age', 'Addicted_Score', 'Conflicts_Over_Social_Media']
    X = df[features]
    y = df['Relationship_Status']

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    resultado = None
    datos_usuario = {}
    grafica_img = None

    if request.method == 'POST':
        try:
            datos_usuario = {
                'Gender': int(request.form['genero']),
                'Age': float(request.form['edad']),
                'Addicted_Score': float(request.form['adiccion']),
                'Conflicts_Over_Social_Media': int(request.form['conflictos'])
            }
            entrada = pd.DataFrame([datos_usuario])
            probs = modelo.predict_proba(entrada)[0]
            pred = modelo.predict(entrada)[0]
            resultado = le_status.inverse_transform([pred])[0]

            clases = le_status.inverse_transform(modelo.classes_)
        except Exception as e:
            resultado = f"Error en la predicción: {str(e)}"

        fig, ax = plt.subplots()
        ax.bar(clases, probs, color="#ec4899")
        ax.set_ylim(0, 1)
        ax.set_title("Probabilidad por Estado de Relación")
        ax.set_ylabel("Probabilidad")
        ax.set_xlabel("Estado")
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        grafica_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

    return render_template("relacion_estado_dashboard.html",
                           resultado=resultado,
                           datos_usuario=datos_usuario,
                           grafica_img=grafica_img)

@app.route('/red_neuronal', methods=['GET', 'POST'])
def red_neuronal_dashboard():
    df = pd.read_csv("Students Social Media Addiction.csv")

    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    def clasificar_adiccion(valor):
        if valor < 4:
            return 0
        elif valor < 7:
            return 1
        else:
            return 2

    df['Nivel_Adiccion'] = df['Addicted_Score'].apply(clasificar_adiccion)

    features = ['Avg_Daily_Usage_Hours', 'Age', 'Gender', 'Sleep_Hours_Per_Night']
    X = df[features]
    y = df['Nivel_Adiccion']

    modelo = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    modelo.fit(X, y)

    resultado = None
    datos_usuario = {}
    grafica_img = None

    niveles = {0: "Bajo", 1: "Medio", 2: "Alto"}

    if request.method == 'POST':
        try:
            datos_usuario = {
                'Avg_Daily_Usage_Hours': float(request.form['horas']),
                'Age': float(request.form['edad']),
                'Gender': int(request.form['genero']),
                'Sleep_Hours_Per_Night': float(request.form['sueño'])
            }
            entrada = pd.DataFrame([datos_usuario])
            pred = modelo.predict(entrada)[0]
            resultado = f"Nivel de adicción: {niveles[pred]}"

            probabilidades = modelo.predict_proba(entrada)[0]

            fig, ax = plt.subplots()
            ax.bar(niveles.values(), probabilidades, color=["#34d399", "#fbbf24", "#ef4444"])
            ax.set_ylim(0, 1)
            ax.set_title("Probabilidad por Nivel de Adicción")
            ax.set_ylabel("Probabilidad")
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            grafica_img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        except Exception as e:
            resultado = f"Error: {str(e)}"

    return render_template("red_neuronal_dashboard.html",
                           resultado=resultado,
                           datos_usuario=datos_usuario,
                           grafica_img=grafica_img)


if __name__ == '__main__':
    app.run(debug=True)
