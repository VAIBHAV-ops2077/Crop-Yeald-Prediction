// main.js — handles form submit and result display

const form    = document.getElementById('predict-form');
const btn     = document.getElementById('predict-btn');

const idle    = document.getElementById('result-idle');
const loading = document.getElementById('result-loading');
const success = document.getElementById('result-success');
const error   = document.getElementById('result-error');

function showState(state) {
  [idle, loading, success, error].forEach(el => el.classList.add('hidden'));
  state.classList.remove('hidden');
}

form.addEventListener('submit', async function (e) {
  e.preventDefault();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = {
    avg_temperature_c:      parseFloat(document.getElementById('avg_temperature_c').value),
    annual_rainfall_mm:     parseFloat(document.getElementById('annual_rainfall_mm').value),
    fertilizer_kg_per_ha:   parseFloat(document.getElementById('fertilizer_kg_per_ha').value),
    pesticide_litre_per_ha: parseFloat(document.getElementById('pesticide_litre_per_ha').value),
    soil_moisture_percent:  parseFloat(document.getElementById('soil_moisture_percent').value),
    climate_stress_index:   parseFloat(document.getElementById('climate_stress_index').value),
    crop:                   document.getElementById('crop').value,
    soil_type:              document.getElementById('soil_type').value,
    irrigation_type:        document.getElementById('irrigation_type').value,
    region:                 document.getElementById('region').value,
    farm_type:              document.getElementById('farm_type').value,
  };

  btn.disabled = true;
  showState(loading);

  try {
    const response = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Server returned an error.');
    }

    document.getElementById('result-yield').textContent = data.yield.toFixed(3);
    document.getElementById('range-text').textContent =
      `${data.error_low.toFixed(3)} – ${data.error_high.toFixed(3)} tons/ha`;
    document.getElementById('meta-model').textContent = data.model_name;
    document.getElementById('meta-r2').textContent    = data.model_r2.toFixed(4);
    document.getElementById('meta-rmse').textContent  = `±${data.model_rmse} tons/ha`;

    showState(success);

  } catch (err) {
    document.getElementById('error-message').textContent = err.message;
    showState(error);
  } finally {
    btn.disabled = false;
  }
});
