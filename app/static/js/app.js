const form = document.querySelector('#prediction-form');
const placeholder = document.querySelector('#placeholder');
const result = document.querySelector('#result');
const errorBox = document.querySelector('#error');
const price = document.querySelector('#price');
const crorePrice = document.querySelector('#crore-price');
const range = document.querySelector('#range');
const button = form?.querySelector('button');

const money = value => `PKR ${new Intl.NumberFormat('en-PK', {maximumFractionDigits: 0}).format(value)}`;
const crore = value => `${(value / 10000000).toFixed(2)} crore PKR`;
const checkboxNames = [
    'furnished', 'gym', 'study_room', 'drawing_room', 'dining_room', 'lawn_garden',
    'swimming_pool', 'electricity_backup', 'lounge_sitting_room', 'corner', 'park_facing',
    'basement', 'brand_new', 'double_unit', 'under_construction'
];

form?.addEventListener('submit', async event => {
    event.preventDefault();
    errorBox.classList.add('hidden');
    button.disabled = true;
    button.querySelector('span').textContent = 'Calculating...';
    const payload = Object.fromEntries(new FormData(form).entries());
    checkboxNames.forEach(name => payload[name] = payload[name] || 0);
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Prediction failed.');
        price.textContent = money(data.price_pkr);
        crorePrice.textContent = crore(data.price_pkr);
        range.textContent = `${money(data.lower_90)} — ${money(data.upper_90)}`;
        placeholder.classList.add('hidden');
        result.classList.remove('hidden');
    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove('hidden');
    } finally {
        button.disabled = false;
        button.querySelector('span').textContent = 'Calculate estimated value';
    }
});
