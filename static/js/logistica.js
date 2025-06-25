async function cargarDatosLogistica() {
  const res = await fetch('/api/logistica');
  const data = await res.json();

  const reales = data.reales;
  const predicciones = data.predicciones;
  const curva = data.curva_logistica;  // <- nueva curva

  const ctx = document.getElementById('graficoLogistico').getContext('2d');
  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Datos Reales',
          data: reales,
          backgroundColor: 'blue'
        },
        {
          label: 'Predicciones',
          data: predicciones,
          backgroundColor: 'red',
          pointStyle: 'cross'
        },
        {
          type: 'line',
          label: 'Curva Logística',
          data: curva,
          borderColor: 'green',
          borderWidth: 2,
          fill: false,
          tension: 0.4, // suaviza la línea
          pointRadius: 0 // oculta los puntos
        }
      ]
    },
    options: {
      scales: {
        x: {
          title: {
            display: true,
            text: 'Horas diarias de uso'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Afecta rendimiento académico'
          },
          ticks: {
            stepSize: 1,
            callback: value => value === 1 ? 'Sí' : 'No'
          },
          min: -0.1,
          max: 1.1
        }
      }
    }
  });
}

cargarDatosLogistica();
