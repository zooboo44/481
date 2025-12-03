async function submitSymptoms() {
    const input = document.getElementById('symptom-input');
    const text = input.value.trim();
    
    if (!text) {
        showError('Please describe your symptoms before submitting.');
        return;
    }
    
    // Hide previous results and errors
    hideAll();
    showLoading();
    
    // Disable submit button
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = true;
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'An error occurred');
        }
        
        displayResults(data);
        
    } catch (error) {
        showError(error.message || 'Failed to get diagnosis. Please try again.');
    } finally {
        hideLoading();
        submitBtn.disabled = false;
    }
}

function displayResults(data) {
    // Show best diagnosis
    document.getElementById('best-disease').textContent = data.best_disease;
    document.getElementById('best-confidence').textContent = 
        `Confidence: ${(data.best_prob * 100).toFixed(1)}%`;
    
    // Show other conditions
    const otherList = document.getElementById('other-conditions-list');
    otherList.innerHTML = '';
    
    data.top_suggestions.forEach(([disease, prob]) => {
        if (disease === data.best_disease) return; // Skip the best one
        
        const li = document.createElement('li');
        li.innerHTML = `
            <span class="condition-name">${disease}</span>
            <span class="condition-prob">${(prob * 100).toFixed(1)}%</span>
        `;
        otherList.appendChild(li);
    });
    
    // Show extracted features
    const featuresDiv = document.getElementById('extracted-features');
    featuresDiv.innerHTML = '';
    
    const featureLabels = {
        'Fever': 'Fever',
        'Cough': 'Cough',
        'Fatigue': 'Fatigue',
        'Difficulty Breathing': 'Difficulty Breathing',
        'Age': 'Age',
        'Gender': 'Gender',
        'Blood Pressure': 'Blood Pressure',
        'Cholesterol Level': 'Cholesterol Level'
    };
    
    Object.entries(data.used_features).forEach(([key, value]) => {
        const featureItem = document.createElement('div');
        featureItem.className = 'feature-item';
        featureItem.innerHTML = `
            <span class="feature-label">${featureLabels[key] || key}:</span>
            <span class="feature-value">${value}</span>
        `;
        featuresDiv.appendChild(featureItem);
    });
    
    // Show results section
    document.getElementById('results').classList.remove('hidden');
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearResults() {
    document.getElementById('symptom-input').value = '';
    hideAll();
}

function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideAll() {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');
}

// Allow Enter key to submit (Ctrl+Enter or Cmd+Enter)
document.getElementById('symptom-input').addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        submitSymptoms();
    }
});

