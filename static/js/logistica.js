async function cargarDatosLogistica() {
  const res = await fetch('/api/logistica');
  const data = await res.json();

  const reales = data.reales;
  const predicciones = data.predicciones;

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
