async function cargarDatosRegresion() {
    const res = await fetch('/api/regresion');
    const data = await res.json();

    const ctx = document.getElementById('grafico').getContext('2d');
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Datos Reales',
                    data: data.reales,
                    backgroundColor: 'blue'
                },
                {
                    label: 'Línea de Regresión',
                    data: data.prediccion,
                    type: 'line',
                    borderColor: 'red',
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Horas de Uso' } },
                y: { title: { display: true, text: 'Salud Mental' } }
            }
        }
    });
}

cargarDatosRegresion();
