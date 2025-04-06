import { models } from './models.js';

const modelList = document.getElementById('modelList');
const countries = ['NG', 'KE', 'ZA'];

function clearMap() {
  countries.forEach((code) => {
    const el = document.getElementById(code);
    if (el) el.setAttribute('fill', '#ccc');
  });
}

models.forEach((model, index) => {
  const div = document.createElement('div');
  div.className = 'bg-white p-4 rounded shadow cursor-pointer hover:bg-gray-100 transition';
  div.innerHTML = `
    <h2 class="text-xl font-semibold">${model.name}</h2>
    <p class="text-sm text-gray-600">Contributors: ${model.contributorCount} (${model.contributors.join(', ')})</p>
  `;
  div.addEventListener('click', () => {
    clearMap();
    model.contributors.forEach((country) => {
      const code = {
        Nigeria: 'NG',
        Kenya: 'KE',
        'South Africa': 'ZA',
      }[country];

      const el = document.getElementById(code);
      if (el) el.setAttribute('fill', '#34d399'); // green highlight
    });
  });
  modelList.appendChild(div);
});