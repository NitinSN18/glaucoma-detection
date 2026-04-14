// ========== GLOBAL STATE ==========

const globalState = {
    results: [],
    batchFiles: [],
    currentAnalysis: null
};

// ========== INITIALIZATION ==========

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkServerHealth();
});

// ========== EVENT LISTENERS SETUP ==========

function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.addEventListener('click', switchTab);
    });

    // Single image upload
    const singleUploadArea = document.getElementById('single-upload-area');
    singleUploadArea.addEventListener('click', () => {
        document.getElementById('single-image-input').click();
    });
    singleUploadArea.addEventListener('dragover', handleDragOver);
    singleUploadArea.addEventListener('dragleave', handleDragLeave);
    singleUploadArea.addEventListener('drop', (e) => handleDrop(e, 'single'));

    document.getElementById('single-image-input').addEventListener('change', (e) => {
        if (e.target.files[0]) {
            previewSingleImage(e.target.files[0]);
        }
    });

    // Batch upload
    const batchUploadArea = document.getElementById('batch-upload-area');
    batchUploadArea.addEventListener('click', () => {
        document.getElementById('batch-image-input').click();
    });
    batchUploadArea.addEventListener('dragover', handleDragOver);
    batchUploadArea.addEventListener('dragleave', handleDragLeave);
    batchUploadArea.addEventListener('drop', (e) => handleDrop(e, 'batch'));

    document.getElementById('batch-image-input').addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            addBatchFiles(Array.from(e.target.files));
        }
    });
}

// ========== TAB SWITCHING ==========

function switchTab(e) {
    const tabName = e.target.getAttribute('data-tab');
    
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    e.target.classList.add('active');

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');

    // Load history when history tab is clicked
    if (tabName === 'history') {
        displayHistory();
    }
}

// ========== DRAG AND DROP ==========

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e, type) {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.classList.remove('dragover');

    const files = Array.from(e.dataTransfer.files).filter(f => isImageFile(f));
    
    if (type === 'single' && files.length > 0) {
        previewSingleImage(files[0]);
    } else if (type === 'batch' && files.length > 0) {
        addBatchFiles(files);
    }
}

function isImageFile(file) {
    return file.type.startsWith('image/');
}

// ========== SINGLE IMAGE HANDLING ==========

function previewSingleImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = document.getElementById('single-preview-img');
        img.src = e.target.result;
        
        document.getElementById('single-preview').classList.remove('hidden');
        document.getElementById('results-card').style.display = 'none';
    };
    reader.readAsDataURL(file);
    
    globalState.currentFile = file;
}

function resetSingleImage() {
    document.getElementById('single-image-input').value = '';
    document.getElementById('single-preview').classList.add('hidden');
    document.getElementById('results-card').style.display = 'none';
    globalState.currentFile = null;
}

async function analyzeSingleImage() {
    if (!globalState.currentFile) {
        alert('Please select an image first');
        return;
    }

    const mode = document.querySelector('input[name="analysis-mode"]:checked').value;
    
    document.getElementById('single-loading').classList.remove('hidden');
    document.getElementById('analyze-btn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', globalState.currentFile);

        let endpoint = '/api/combined';
        if (mode === 'classify-only') endpoint = '/api/classify';
        if (mode === 'segment-only') endpoint = '/api/segment';

        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Analysis failed');
        }

        // Store result for history
        globalState.currentAnalysis = result;
        globalState.results.push({
            ...result,
            filename: globalState.currentFile.name,
            mode: mode,
            timestamp: new Date(result.timestamp)
        });

        displaySingleResults(result, mode);
        playNotificationSound();

    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image: ' + error.message);
    } finally {
        document.getElementById('single-loading').classList.add('hidden');
        document.getElementById('analyze-btn').disabled = false;
    }
}

function displaySingleResults(result, mode) {
    const resultsCard = document.getElementById('results-card');
    resultsCard.style.display = 'block';

    // Show classification results
    if (mode === 'combined' || mode === 'classify-only') {
        const classifDiv = document.getElementById('classification-results');
        classifDiv.style.display = 'block';

        const confidence = result.confidence * 100;
        const prediction = result.prediction.toLowerCase();
        
        document.getElementById('result-prediction').textContent = prediction.toUpperCase();
        document.getElementById('result-prediction').className = `value ${prediction}`;
        document.getElementById('result-confidence').textContent = confidence.toFixed(1) + '%';
        document.getElementById('confidence-fill').style.width = confidence + '%';
        document.getElementById('prob-glaucoma').textContent = (result.probabilities.glaucoma * 100).toFixed(1) + '%';
        document.getElementById('prob-normal').textContent = (result.probabilities.normal * 100).toFixed(1) + '%';
    } else {
        document.getElementById('classification-results').style.display = 'none';
    }

    // Show segmentation results
    if (mode === 'combined' || mode === 'segment-only') {
        const segDiv = document.getElementById('segmentation-results');
        segDiv.style.display = 'block';

        document.getElementById('result-original').src = 'data:image/png;base64,' + result.original_image;
        document.getElementById('result-overlay').src = 'data:image/png;base64,' + result.overlay;
        document.getElementById('result-disc-mask').src = 'data:image/png;base64,' + result.disc_mask;
        document.getElementById('result-cup-mask').src = 'data:image/png;base64,' + result.cup_mask;
    } else {
        document.getElementById('segmentation-results').style.display = 'none';
    }

    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.getElementById('download-results-btn')?.addEventListener('click', downloadSingleResults);

function downloadSingleResults() {
    if (!globalState.currentAnalysis) return;

    const result = globalState.currentAnalysis;
    const timestamp = new Date(result.timestamp).toLocaleString();
    
    let csvContent = 'Glaucoma Detection Analysis Results\n';
    csvContent += `Timestamp,${timestamp}\n`;
    csvContent += `File,${globalState.currentFile.name}\n\n`;
    csvContent += 'Classification Results\n';
    csvContent += `Prediction,${result.prediction}\n`;
    csvContent += `Confidence,${(result.confidence * 100).toFixed(2)}%\n`;
    csvContent += `Glaucoma Probability,${(result.probabilities.glaucoma * 100).toFixed(2)}%\n`;
    csvContent += `Normal Probability,${(result.probabilities.normal * 100).toFixed(2)}%\n`;

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `glaucoma_results_${Date.now()}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
}

// ========== BATCH PROCESSING ==========

function addBatchFiles(files) {
    globalState.batchFiles = files;
    updateBatchFileList();
}

function updateBatchFileList() {
    const fileList = document.getElementById('batch-file-list');
    const ul = document.getElementById('batch-files');
    const count = document.getElementById('batch-file-count');

    if (globalState.batchFiles.length === 0) {
        fileList.classList.add('hidden');
        document.getElementById('batch-process-btn').style.display = 'none';
        document.getElementById('batch-clear-btn').style.display = 'none';
        return;
    }

    fileList.classList.remove('hidden');
    document.getElementById('batch-process-btn').style.display = 'block';
    document.getElementById('batch-clear-btn').style.display = 'block';

    ul.innerHTML = '';
    globalState.batchFiles.forEach(file => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>${file.name}</span>
            <span class="size">${formatFileSize(file.size)}</span>
        `;
        ul.appendChild(li);
    });

    count.textContent = globalState.batchFiles.length;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function resetBatch() {
    document.getElementById('batch-image-input').value = '';
    globalState.batchFiles = [];
    updateBatchFileList();
    document.getElementById('batch-results-card').style.display = 'none';
    document.getElementById('batch-results-list').innerHTML = '';
}

async function processBatch() {
    if (globalState.batchFiles.length === 0) {
        alert('Please select images first');
        return;
    }

    document.getElementById('batch-loading').classList.remove('hidden');
    document.getElementById('batch-process-btn').disabled = true;
    document.getElementById('batch-results-card').style.display = 'none';

    try {
        const formData = new FormData();
        globalState.batchFiles.forEach(file => {
            formData.append('images', file);
        });

        const response = await fetch('/api/batch', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || 'Batch processing failed');
        }

        displayBatchResults(result);
        playNotificationSound();

    } catch (error) {
        console.error('Error:', error);
        alert('Error processing batch: ' + error.message);
    } finally {
        document.getElementById('batch-loading').classList.add('hidden');
        document.getElementById('batch-process-btn').disabled = false;
    }
}

function displayBatchResults(batchResult) {
    const resultsCard = document.getElementById('batch-results-card');
    const resultsList = document.getElementById('batch-results-list');

    resultsCard.style.display = 'block';
    resultsList.innerHTML = '';

    const { total, successful, results } = batchResult;
    document.getElementById('batch-progress-text').textContent = `${successful} of ${total} processed`;
    document.getElementById('batch-success-count').textContent = `(${successful} successful)`;
    document.getElementById('batch-progress-fill').style.width = (successful / total * 100) + '%';

    results.forEach(itemResult => {
        const div = document.createElement('div');
        div.className = `batch-result-item ${itemResult.success ? 'success' : 'error'}`;

        if (itemResult.success) {
            const confidence = (itemResult.confidence * 100).toFixed(1);
            div.innerHTML = `
                <h4>📄 ${itemResult.filename}</h4>
                <img src="data:image/png;base64,${itemResult.overlay}" alt="Result">
                <div class="batch-result-prediction">
                    <span class="label">Diagnosis:</span>
                    <span class="value">${itemResult.prediction.toUpperCase()}</span>
                </div>
                <div class="batch-result-prediction" style="background-color: #f3f4f6;">
                    <span class="label">Confidence: ${confidence}%</span>
                </div>
            `;
            
            // Store in results for history
            globalState.results.push({
                ...itemResult,
                timestamp: new Date(itemResult.timestamp),
                mode: 'combined'
            });
        } else {
            div.innerHTML = `
                <h4>❌ ${itemResult.filename}</h4>
                <p class="batch-result-error">${itemResult.error}</p>
            `;
        }

        resultsList.appendChild(div);
    });

    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function downloadBatchResults() {
    // This could be enhanced to include all batch results
    alert('Batch results CSV export coming soon');
}

// ========== HISTORY ==========

function displayHistory() {
    const container = document.getElementById('history-container');
    const clearBtn = document.getElementById('clear-history-btn');

    if (globalState.results.length === 0) {
        container.innerHTML = '<p style="color: #666;">No results yet. Analyze an image to get started.</p>';
        clearBtn.style.display = 'none';
        return;
    }

    clearBtn.style.display = 'block';
    container.innerHTML = '';

    globalState.results.forEach((result, index) => {
        const div = document.createElement('div');
        div.className = 'history-item';
        
        const timestamp = result.timestamp instanceof Date 
            ? result.timestamp.toLocaleString()
            : new Date(result.timestamp).toLocaleString();

        div.innerHTML = `
            <img src="data:image/png;base64,${result.overlay || result.original_image}" alt="${result.filename}">
            <div class="history-details">
                <h4>${result.filename}</h4>
                <p>Mode: ${result.mode}</p>
                <p>${timestamp}</p>
            </div>
            <div class="history-result">
                <div class="prediction">${result.prediction ? result.prediction.toUpperCase() : 'N/A'}</div>
                <div class="confidence">
                    ${result.confidence ? (result.confidence * 100).toFixed(1) + '%' : ''}
                </div>
            </div>
        `;

        container.appendChild(div);
    });
}

function clearHistory() {
    if (confirm('Are you sure you want to clear all history?')) {
        globalState.results = [];
        displayHistory();
    }
}

// ========== SERVER HEALTH CHECK ==========

async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');

        if (data.status === 'ok') {
            indicator.classList.add('healthy');
            text.textContent = `Ready • Device: ${data.device}`;
        } else {
            text.textContent = 'Error: Server not responsive';
        }
    } catch (error) {
        console.error('Health check failed:', error);
        document.getElementById('status-text').textContent = 'Error: Cannot connect to server';
    }
}

// ========== UTILITIES ==========

function playNotificationSound() {
    try {
        const audio = document.getElementById('notification-sound');
        if (audio) {
            audio.play().catch(() => {
                // Silently fail if audio cannot play
            });
        }
    } catch (e) {
        // Silently ignore audio errors
    }
}
