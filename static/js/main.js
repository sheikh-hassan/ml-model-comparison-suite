// Main JavaScript for Assignment 2

document.addEventListener('DOMContentLoaded', function() {
    console.log('Assignment 2 - ML Algorithms Comparison loaded');
});

// Utility functions (same as Assignment 1)
async function fetchAPI(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Format percentage
function formatPercent(value) {
    return (value * 100).toFixed(2) + '%';
}

// Format number to 4 decimal places
function formatMetric(value) {
    return parseFloat(value).toFixed(4);
}

// Show notification
function showNotification(message, type = 'info') {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    alert.style.position = 'fixed';
    alert.style.top = '20px';
    alert.style.right = '20px';
    alert.style.zIndex = '1000';
    alert.style.minWidth = '300px';
    
    document.body.appendChild(alert);
    
    setTimeout(() => {
        alert.remove();
    }, 5000);
}

// Predict with all algorithms
async function predictAllCSV(data) {
    try {
        const result = await fetchAPI('/predict/all/csv', {
            method: 'POST',
            body: JSON.stringify(data)
        });
        
        if (result.success) {
            showNotification('Predictions from all algorithms received', 'success');
            return result.predictions;
        } else {
            showNotification(`Error: ${result.error}`, 'error');
            return null;
        }
    } catch (error) {
        showNotification('Failed to make predictions', 'error');
        return null;
    }
}

// Predict MNIST with all algorithms
async function predictAllMNIST(imageFile) {
    try {
        const formData = new FormData();
        formData.append('image', imageFile);
        
        const response = await fetch('/predict/all/mnist', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Predictions from all algorithms received', 'success');
            return result.predictions;
        } else {
            showNotification(`Error: ${result.error}`, 'error');
            return null;
        }
    } catch (error) {
        showNotification('Failed to make predictions', 'error');
        return null;
    }
}

// Get available algorithms
async function getAlgorithms() {
    try {
        return await fetchAPI('/api/algorithms');
    } catch (error) {
        console.error('Error getting algorithms:', error);
        return null;
    }
}

// Load model info
async function loadModelInfo() {
    try {
        const response = await fetch('/test');
        const data = await response.json();
        console.log('Model Status:', data);
        return data;
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Compare results
function compareResults(predictions) {
    let html = '<table class="comparison-table"><thead><tr><th>Algorithm</th><th>Prediction</th><th>Confidence</th></tr></thead><tbody>';
    
    for (const [algo, result] of Object.entries(predictions)) {
        html += `<tr>
            <td><strong>${algo}</strong></td>
            <td>${result.prediction}</td>
            <td>${(result.confidence * 100).toFixed(2)}%</td>
        </tr>`;
    }
    
    html += '</tbody></table>';
    return html;
}

function parseCsvRow(text) {
    const lines = text.trim().split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) {
        throw new Error('CSV must include a header row and one data row');
    }

    const headers = lines[0].split(',').map(value => value.trim());
    const values = lines[1].split(',').map(value => value.trim());
    const record = {};

    headers.forEach((header, index) => {
        record[header] = values[index];
    });

    return record;
}

function renderInlineResult(elementId, title, lines, tone = 'info') {
    const element = document.getElementById(elementId);
    if (!element) {
        return;
    }

    element.className = `result-box result-${tone}`;
    element.innerHTML = `<strong>${title}</strong><br>${lines.join('<br>')}`;
}

async function handleCsvFilePrediction() {
    const fileInput = document.getElementById('csv-file-input');
    const algoSelect = document.getElementById('csv-algorithm-select');
    const file = fileInput?.files?.[0];

    if (!file) {
        showNotification('Choose a CSV file first', 'error');
        return;
    }

    try {
        const text = await file.text();
        const record = parseCsvRow(text);
        const payload = {
            Pclass: Number(record.Pclass),
            Age: Number(record.Age),
            SibSp: Number(record.SibSp),
            Parch: Number(record.Parch),
            Fare: Number(record.Fare)
        };

        const algorithm = algoSelect?.value || 'KNN';
        const result = await fetchAPI(`/predict/csv/${encodeURIComponent(algorithm)}`, {
            method: 'POST',
            body: JSON.stringify(payload)
        });

        if (result.success) {
            renderInlineResult('csv-file-result', 'Prediction Ready', [
                `Algorithm: ${result.algorithm}`,
                `Prediction: ${result.prediction_label}`,
                `Confidence: ${(result.confidence * 100).toFixed(2)}%`
            ], 'success');
        } else {
            renderInlineResult('csv-file-result', 'Prediction Failed', [result.error], 'danger');
        }
    } catch (error) {
        renderInlineResult('csv-file-result', 'Upload Error', [error.message], 'danger');
    }
}

async function handleMnistFilePrediction() {
    const fileInput = document.getElementById('mnist-file-input');
    const algoSelect = document.getElementById('mnist-algorithm-select');
    const file = fileInput?.files?.[0];

    if (!file) {
        showNotification('Choose an image first', 'error');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('image', file);

        const algorithm = algoSelect?.value || 'KNN';
        const response = await fetch(`/predict/mnist/${encodeURIComponent(algorithm)}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            renderInlineResult('mnist-file-result', 'Prediction Ready', [
                `Algorithm: ${result.algorithm}`,
                `Predicted Digit: ${result.prediction}`,
                `Confidence: ${(result.confidence * 100).toFixed(2)}%`
            ], 'success');
        } else {
            renderInlineResult('mnist-file-result', 'Prediction Failed', [result.error], 'danger');
        }
    } catch (error) {
        renderInlineResult('mnist-file-result', 'Prediction Error', [error.message], 'danger');
    }
}

document.addEventListener('DOMContentLoaded', function () {
    const csvButton = document.getElementById('csv-file-btn');
    const mnistButton = document.getElementById('mnist-file-btn');

    if (csvButton) {
        csvButton.addEventListener('click', handleCsvFilePrediction);
    }

    if (mnistButton) {
        mnistButton.addEventListener('click', handleMnistFilePrediction);
    }
});
