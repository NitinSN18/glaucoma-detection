// ========== GLOBAL STATE ==========

const globalState = {
  results: [],
  batchFiles: [],
  currentAnalysis: null,
  currentFile: null,
  currentPatientDetails: null
};

const DISCLAIMER_KEY = 'gds-disclaimer-accepted-v1';

const workplaceFlow = {
  input: {
    title: 'Input Quality Gate',
    description:
        'A clean fundus image is the strongest predictor of stable downstream output. We check framing, focus, and glare before inference.',
    metrics: [
      {label: 'Target Resolution', value: '512 x 512'},
      {label: 'Accepted Formats', value: 'PNG/JPG/BMP'},
      {label: 'Quality Threshold', value: '>= 0.80'}
    ],
    points: [
      'Centered optic nerve head improves segmentation stability.',
      'Low glare and low blur reduce false-positive risk spikes.',
      'Extreme underexposure can suppress vessel contrast.'
    ]
  },
  preprocess: {
    title: 'Preprocess & Normalize',
    description:
        'Input is resized and normalized to match model training distributions so feature extraction remains consistent across devices.',
    metrics: [
      {label: 'Color Normalization', value: 'Enabled'},
      {label: 'Resize Kernel', value: 'Bilinear'},
      {label: 'Pipeline Latency', value: '~45 ms'}
    ],
    points: [
      'Normalization reduces variability from camera and clinic lighting.',
      'Consistent sizing keeps receptive fields comparable to training.',
      'Preprocess timing remains low for near-real-time use.'
    ]
  },
  classify: {
    title: 'Classification: EfficientNet-B0',
    description:
        'EfficientNet-B0 predicts glaucoma probability using global retinal cues and reports calibrated confidence for triage decisions.',
    metrics: [
      {label: 'Backbone', value: 'EfficientNet-B0'},
      {label: 'Output Classes', value: '2 (Glaucoma/Normal)'},
      {label: 'Typical Inference', value: '~120 ms'}
    ],
    points: [
      'Probability scores are shown for both glaucoma and normal.',
      'Confidence bar highlights certainty for review prioritization.',
      'Use with clinical context rather than standalone diagnosis.'
    ]
  },
  segment: {
    title: 'Segmentation: DeepLabV3+',
    description:
        'DeepLabV3+ extracts optic disc and cup masks to provide structural evidence and CDR-relevant visual cues.',
    metrics: [
      {label: 'Model', value: 'DeepLabV3+'},
      {label: 'Outputs', value: 'Disc + Cup masks'},
      {label: 'Overlay Mode', value: 'Color-coded'}
    ],
    points: [
      'Disc and cup boundaries are rendered as separate overlays.',
      'Quality warnings suppress unsafe structural metrics when needed.',
      'Heatmaps support explainability and confidence review.'
    ]
  },
  report: {
    title: 'Integrated Clinical Report',
    description:
        'Classification and segmentation are merged into one compact report for interpretation, handoff, and follow-up tracking.',
    metrics: [
      {label: 'Includes', value: 'Scores + Overlays'},
      {label: 'Snapshot', value: 'Patient summary'},
      {label: 'Export', value: 'CSV result download'}
    ],
    points: [
      'Patient context is attached to each analysis run when selected.',
      'Clinical recommendation text is surfaced for quick actioning.',
      'Results can be reviewed later in history and logbook tabs.'
    ]
  }
};

const workplaceOrder = ['input', 'preprocess', 'classify', 'segment', 'report'];

// ========== INITIALIZATION ==========

document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  setupDisplayControls();
  setupWorkplaceFlowchart();
  setupSettingsModal();
  setupDisclaimerPopup();
  startSplashSequence();
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
  const singleInput = document.getElementById('single-image-input');
  if (singleUploadArea && singleInput) {
    singleUploadArea.addEventListener('click', () => {
      singleInput.click();
    });
    singleUploadArea.addEventListener('dragover', handleDragOver);
    singleUploadArea.addEventListener('dragleave', handleDragLeave);
    singleUploadArea.addEventListener('drop', (e) => handleDrop(e, 'single'));

    singleInput.addEventListener('change', (e) => {
      if (e.target.files[0]) {
        previewSingleImage(e.target.files[0]);
      }
    });
  }

  // Batch upload
  const batchUploadArea = document.getElementById('batch-upload-area');
  const batchInput = document.getElementById('batch-image-input');
  if (batchUploadArea && batchInput) {
    batchUploadArea.addEventListener('click', () => {
      batchInput.click();
    });
    batchUploadArea.addEventListener('dragover', handleDragOver);
    batchUploadArea.addEventListener('dragleave', handleDragLeave);
    batchUploadArea.addEventListener('drop', (e) => handleDrop(e, 'batch'));

    batchInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        addBatchFiles(Array.from(e.target.files));
      }
    });
  }

  document.querySelectorAll('input[name="analysis-mode"]').forEach(input => {
    input.addEventListener('change', updateCurrentWorkflowCard);
  });

  document.getElementById('save-patient-btn')
      ?.addEventListener('click', savePatientToLogbook);
  document.getElementById('refresh-logbook-btn')
      ?.addEventListener('click', loadPatientLogbook);

  document.getElementById('to-upload-step-btn')
      ?.addEventListener('click', () => setSingleAnalysisStep('upload'));
  document.getElementById('to-patient-step-btn')
      ?.addEventListener('click', () => setSingleAnalysisStep('patient'));
}

function setupSettingsModal() {
  const settingsBtn = document.getElementById('settings-btn');
  const settingsModal = document.getElementById('settings-modal');
  const settingsClose = document.getElementById('settings-close');
  const logoutBtn = document.getElementById('logout-btn');

  if (settingsBtn && settingsModal) {
    settingsBtn.addEventListener('click', () => {
      settingsModal.style.display = 'flex';
    });
  }

  if (settingsClose && settingsModal) {
    settingsClose.addEventListener('click', () => {
      settingsModal.style.display = 'none';
    });
  }

  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      try {
        await fetch('/api/logout', {method: 'POST'});
      } catch (_) {
        // Ignore network errors and still redirect.
      }
      window.location.href = '/login';
    });
  }
}

function setupDisclaimerPopup() {
  const modal = document.getElementById('disclaimer-modal');
  const closeBtn = document.getElementById('disclaimer-close-btn');
  if (!modal || !closeBtn) {
    return;
  }

  const alreadyAccepted = window.localStorage.getItem(DISCLAIMER_KEY) === '1';
  if (!alreadyAccepted) {
    modal.style.display = 'flex';
  }

  closeBtn.addEventListener('click', () => {
    window.localStorage.setItem(DISCLAIMER_KEY, '1');
    modal.style.display = 'none';
  });
}

function setupDisplayControls() {
  const themeSelect = document.getElementById('visual-mode-select');

  if (themeSelect) {
    const savedTheme = window.localStorage.getItem('visual-mode') || 'night';
    themeSelect.value = savedTheme;
    document.body.dataset.theme = savedTheme;
    themeSelect.addEventListener('change', () => {
      document.body.dataset.theme = themeSelect.value;
      window.localStorage.setItem('visual-mode', themeSelect.value);
    });
  }

  updateCurrentWorkflowCard();
}

function setupWorkplaceFlowchart() {
  const nodes = document.querySelectorAll('#model-flowchart .flow-step');
  const title = document.getElementById('workplace-slide-title');
  const description = document.getElementById('workplace-slide-summary');
  const metrics = document.getElementById('workplace-slide-metrics');
  const points = document.getElementById('workplace-slide-points');
  const index = document.getElementById('workplace-slide-index');
  const prevBtn = document.getElementById('workplace-prev-btn');
  const nextBtn = document.getElementById('workplace-next-btn');

  if (!nodes.length || !title || !description || !metrics || !points ||
      !index) {
    return;
  }

  let activeIndex = 0;

  const renderSlide = () => {
    const key = workplaceOrder[activeIndex];
    const info = workplaceFlow[key];
    if (!info) return;

    title.textContent = info.title;
    description.textContent = info.description;
    index.textContent = `${activeIndex + 1} / ${workplaceOrder.length}`;

    metrics.innerHTML = '';
    (info.metrics || []).forEach((item) => {
      const metricCard = document.createElement('div');
      metricCard.className = 'ppt-metric-item';
      metricCard.innerHTML =
          `<span>${item.label}</span><strong>${item.value}</strong>`;
      metrics.appendChild(metricCard);
    });

    points.innerHTML = '';
    (info.points || []).forEach((point) => {
      const li = document.createElement('li');
      li.textContent = point;
      points.appendChild(li);
    });

    if (prevBtn) prevBtn.disabled = activeIndex === 0;
    if (nextBtn) nextBtn.disabled = activeIndex === workplaceOrder.length - 1;
  };

  const setActiveNode = (selectedNode) => {
    nodes.forEach(node => node.classList.remove('active'));
    selectedNode.classList.add('active');

    const key = selectedNode.getAttribute('data-model');
    const nextIndex = workplaceOrder.indexOf(key);
    if (nextIndex !== -1) {
      activeIndex = nextIndex;
      renderSlide();
    }
  };

  const setActiveByIndex = (nextIndex) => {
    if (nextIndex < 0 || nextIndex >= workplaceOrder.length) return;
    activeIndex = nextIndex;
    const key = workplaceOrder[activeIndex];
    const matchingNode = Array.from(nodes).find(
        (node) => node.getAttribute('data-model') === key);
    if (matchingNode) {
      nodes.forEach((node) => node.classList.remove('active'));
      matchingNode.classList.add('active');
    }
    renderSlide();
  };

  nodes.forEach(node => {
    node.addEventListener('click', () => setActiveNode(node));
  });

  prevBtn?.addEventListener('click', () => setActiveByIndex(activeIndex - 1));
  nextBtn?.addEventListener('click', () => setActiveByIndex(activeIndex + 1));

  setActiveNode(nodes[0]);
}

function setSingleAnalysisStep(step) {
  const patientStep = document.getElementById('analysis-step-patient');
  const uploadStep = document.getElementById('analysis-step-upload');
  const patientPill = document.getElementById('pill-step-patient');
  const uploadPill = document.getElementById('pill-step-upload');

  if (!patientStep || !uploadStep || !patientPill || !uploadPill) {
    return;
  }

  const showUpload = step === 'upload';
  patientStep.classList.toggle('active', !showUpload);
  uploadStep.classList.toggle('active', showUpload);
  patientPill.classList.toggle('active', !showUpload);
  uploadPill.classList.toggle('active', showUpload);
}

function startSplashSequence() {
  const splash = document.getElementById('startup-splash');
  if (!splash) {
    return;
  }

  window.setTimeout(() => {
    splash.classList.add('hidden');
  }, 1300);
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
  if (tabName === 'logbook') {
    loadPatientLogbook();
  }
}

function updateCurrentWorkflowCard() {
  const workflowCard = document.getElementById('current-workflow-text');
  if (!workflowCard) {
    return;
  }

  const selectedMode =
      document.querySelector('input[name="analysis-mode"]:checked')?.value ||
      'combined';
  const messages = {
    combined:
        'Change mode, upload image, view confidence and segmentation overlay.',
    'classify-only':
        'Classification mode focuses on glaucoma risk and confidence scoring.',
    'segment-only':
        'Segmentation mode highlights the optic disc and cup boundaries.'
  };

  workflowCard.textContent = messages[selectedMode] || messages.combined;
}

function collectPatientDetails() {
  const getValue = (id) => {
    const element = document.getElementById(id);
    return element ? element.value.trim() : '';
  };

  const details = {
    first_name: getValue('patient-first-name'),
    last_name: getValue('patient-last-name'),
    patient_id: getValue('patient-id'),
    age: getValue('patient-age'),
    gender: getValue('patient-gender'),
    email: getValue('patient-email'),
    phone: getValue('patient-phone'),
    eye_examined: getValue('patient-eye'),
    diabetes: getValue('patient-diabetes'),
    hypertension: getValue('patient-hypertension'),
    family_history_glaucoma: getValue('patient-family-history'),
    allergies: getValue('patient-allergies'),
    allergy_details: getValue('patient-allergy-details'),
    medications: getValue('patient-medications'),
    eye_surgery_history: getValue('patient-eye-surgery'),
    chief_complaint: getValue('patient-chief-complaint'),
    clinical_notes: getValue('patient-notes')
  };

  return details;
}

function validatePatientDetails(details) {
  return true;
}

function renderPatientSummary(details) {
  const summaryEl = document.getElementById('patient-summary-text');
  if (!summaryEl) {
    return;
  }

  if (!details) {
    summaryEl.textContent = 'Patient details will appear here after analysis.';
    return;
  }

  const title =
      [details.first_name, details.last_name].filter(Boolean).join(' ');
  const patientId =
      details.patient_id ? `ID: ${details.patient_id}` : 'ID: N/A';
  const demographics = [
    details.age ? `${details.age}y` : '', details.gender
  ].filter(Boolean).join(' | ');
  const risks = [
    details.diabetes === 'yes' ? 'Diabetes' : '',
    details.hypertension === 'yes' ? 'Hypertension' : '',
    details.family_history_glaucoma === 'yes' ? 'Family Hx Glaucoma' : ''
  ].filter(Boolean).join(', ');
  const riskLine = risks || 'No major risk flags captured';
  summaryEl.textContent =
      `${title} | ${patientId} | ${demographics} | ${riskLine}`;
}

async function fetchWithAuth(url, options) {
  const response = await fetch(url, options);
  if (response.status === 401) {
    window.location.href = '/login';
    throw new Error('Session expired. Please log in again.');
  }
  return response;
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
  globalState.currentPatientDetails = null;
  renderPatientSummary(null);
}

async function analyzeSingleImage() {
  if (!globalState.currentFile) {
    alert('Please select an image first');
    return;
  }

  const mode =
      document.querySelector('input[name="analysis-mode"]:checked').value;

  const patientDetails = collectPatientDetails();
  if (!validatePatientDetails(patientDetails)) {
    return;
  }
  globalState.currentPatientDetails = patientDetails;

  document.getElementById('single-loading').classList.remove('hidden');
  document.getElementById('analyze-btn').disabled = true;

  try {
    const formData = new FormData();
    formData.append('image', globalState.currentFile);
    formData.append('patient_details', JSON.stringify(patientDetails));
    formData.append(
        'save_to_logbook',
        document.getElementById('save-logbook-checkbox')?.checked ? 'true' :
                                                                    'false');

    let endpoint = '/api/combined';
    if (mode === 'classify-only') endpoint = '/api/classify';
    if (mode === 'segment-only') endpoint = '/api/segment';

    const response =
        await fetchWithAuth(endpoint, {method: 'POST', body: formData});

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
      timestamp: new Date(result.timestamp),
      patient_details: patientDetails
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

  // Store result in global state for modal access
  globalState.currentAnalysis = result;

  // Show classification results
  if (mode === 'combined' || mode === 'classify-only') {
    const classifDiv = document.getElementById('classification-results');
    classifDiv.style.display = 'block';

    const confidence = result.confidence * 100;
    const prediction = result.prediction.toLowerCase();

    document.getElementById('result-prediction').textContent =
        prediction.toUpperCase();
    document.getElementById('result-prediction').className =
        `value ${prediction}`;
    document.getElementById('result-confidence').textContent =
        confidence.toFixed(1) + '%';
    document.getElementById('confidence-fill').style.width = confidence + '%';
    document.getElementById('prob-glaucoma').textContent =
        (result.probabilities.glaucoma * 100).toFixed(1) + '%';
    document.getElementById('prob-normal').textContent =
        (result.probabilities.normal * 100).toFixed(1) + '%';

    const metrics = result.clinical_metrics || {};
    const valid = metrics.is_valid !== false;
    document.getElementById('inline-sensitivity').textContent =
        valid && metrics.sensitivity != null ? `${metrics.sensitivity}%` :
                                               'N/A';
    document.getElementById('inline-specificity').textContent =
        valid && metrics.specificity != null ? `${metrics.specificity}%` :
                                               'N/A';
    document.getElementById('inline-cdr').textContent =
        valid && typeof metrics.cup_to_disc_ratio === 'number' ?
        metrics.cup_to_disc_ratio.toFixed(3) :
        'N/A';
    document.getElementById('inline-severity').textContent =
        metrics.severity || 'N/A';
    document.getElementById('inline-recommendation-text').textContent =
        metrics.recommendation ||
        'Clinical recommendation appears for combined analysis.';
  } else {
    document.getElementById('classification-results').style.display = 'none';
  }

  // Show segmentation results
  if (mode === 'combined' || mode === 'segment-only') {
    const segDiv = document.getElementById('segmentation-results');
    segDiv.style.display = 'block';

    const qualityWarning =
        document.getElementById('segmentation-quality-warning');
    const quality = result.segmentation_quality || null;
    if (qualityWarning) {
      if (quality && quality.reliable === false) {
        qualityWarning.textContent = `Segmentation quality warning: ${
            quality.reason}. Structural metrics are suppressed for safety.`;
        qualityWarning.classList.remove('hidden');
      } else {
        qualityWarning.textContent = '';
        qualityWarning.classList.add('hidden');
      }
    }

    document.getElementById('result-original').src =
        'data:image/png;base64,' + result.original_image;
    document.getElementById('result-overlay').src =
        'data:image/png;base64,' + result.overlay;
    document.getElementById('result-disc-mask').src =
        'data:image/png;base64,' + result.disc_mask;
    document.getElementById('result-cup-mask').src =
        'data:image/png;base64,' + result.cup_mask;

    const discHeatmapEl = document.getElementById('result-disc-heatmap');
    const cupHeatmapEl = document.getElementById('result-cup-heatmap');
    if (discHeatmapEl && result.disc_heatmap) {
      discHeatmapEl.src = 'data:image/png;base64,' + result.disc_heatmap;
    } else if (discHeatmapEl) {
      discHeatmapEl.src = '';
    }

    if (cupHeatmapEl && result.cup_heatmap) {
      cupHeatmapEl.src = 'data:image/png;base64,' + result.cup_heatmap;
    } else if (cupHeatmapEl) {
      cupHeatmapEl.src = '';
    }
  } else {
    document.getElementById('segmentation-results').style.display = 'none';
  }

  renderPatientSummary(
      result.patient_details || globalState.currentPatientDetails);
  updateCurrentWorkflowCard();

  // Scroll to results
  resultsCard.scrollIntoView({behavior: 'smooth', block: 'start'});
}

document.getElementById('download-results-btn')
    ?.addEventListener('click', downloadSingleResults);

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
  csvContent += `Glaucoma Probability,${
      (result.probabilities.glaucoma * 100).toFixed(2)}%\n`;
  csvContent +=
      `Normal Probability,${(result.probabilities.normal * 100).toFixed(2)}%\n`;

  const blob = new Blob([csvContent], {type: 'text/csv'});
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

    const response =
        await fetchWithAuth('/api/batch', {method: 'POST', body: formData});

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

  const {total, successful, results} = batchResult;
  document.getElementById('batch-progress-text').textContent =
      `${successful} of ${total} processed`;
  document.getElementById('batch-success-count').textContent =
      `(${successful} successful)`;
  document.getElementById('batch-progress-fill').style.width =
      (successful / total * 100) + '%';

  results.forEach(itemResult => {
    const div = document.createElement('div');
    div.className =
        `batch-result-item ${itemResult.success ? 'success' : 'error'}`;

    if (itemResult.success) {
      const confidence = (itemResult.confidence * 100).toFixed(1);
      div.innerHTML = `
            <h4>Image: ${itemResult.filename}</h4>
                <img src="data:image/png;base64,${
          itemResult.overlay}" alt="Result">
                <div class="batch-result-prediction">
                    <span class="label">Diagnosis:</span>
                    <span class="value">${
          itemResult.prediction.toUpperCase()}</span>
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
                <h4>Failed: ${itemResult.filename}</h4>
                <p class="batch-result-error">${itemResult.error}</p>
            `;
    }

    resultsList.appendChild(div);
  });

  resultsCard.scrollIntoView({behavior: 'smooth', block: 'start'});
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
    container.innerHTML =
        '<p style="color: #666;">No results yet. Analyze an image to get started.</p>';
    clearBtn.style.display = 'none';
    return;
  }

  clearBtn.style.display = 'block';
  container.innerHTML = '';

  globalState.results.forEach((result, index) => {
    const div = document.createElement('div');
    div.className = 'history-item';

    const timestamp = result.timestamp instanceof Date ?
        result.timestamp.toLocaleString() :
        new Date(result.timestamp).toLocaleString();

    div.innerHTML = `
            <img src="data:image/png;base64,${
        result.overlay || result.original_image}" alt="${result.filename}">
            <div class="history-details">
                <h4>${result.filename}</h4>
                <p>Mode: ${result.mode}</p>
                <p>${timestamp}</p>
            </div>
            <div class="history-result">
                <div class="prediction">${
        result.prediction ? result.prediction.toUpperCase() : 'N/A'}</div>
                <div class="confidence">
                    ${
        result.confidence ? (result.confidence * 100).toFixed(1) + '%' : ''}
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

async function savePatientToLogbook() {
  const patientDetails = collectPatientDetails();
  const hasAnyField = Object.values(patientDetails).some(value => value);
  if (!hasAnyField) {
    alert('Please enter at least one patient detail before saving to logbook.');
    return;
  }

  try {
    const response = await fetchWithAuth('/api/patient-records', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({patient_details: patientDetails})
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Unable to save patient details');
    }
    alert('Patient details saved to logbook.');
    loadPatientLogbook();
  } catch (error) {
    alert('Error saving patient details: ' + error.message);
  }
}

async function loadPatientLogbook() {
  const container = document.getElementById('logbook-container');
  if (!container) {
    return;
  }

  container.innerHTML = '<p style="color: #9bb3d4;">Loading records...</p>';
  try {
    const response = await fetchWithAuth('/api/patient-records');
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'Unable to load logbook');
    }

    const records =
        Array.isArray(data.records) ? [...data.records].reverse() : [];
    if (!records.length) {
      container.innerHTML =
          '<p style="color: #9bb3d4;">No patient records yet.</p>';
      return;
    }

    container.innerHTML = '';
    records.forEach((record) => {
      const details = record.patient_details || {};
      const card = document.createElement('div');
      card.className = 'logbook-item';
      const timestamp = record.timestamp ?
          new Date(record.timestamp).toLocaleString() :
          'Unknown time';
      const fullName =
          [details.first_name, details.last_name].filter(Boolean).join(' ') ||
          'Unnamed patient';
      const allergies = details.allergies === 'yes' ?
          (details.allergy_details || 'Yes') :
          (details.allergies || 'Unknown');
      const diagnosis =
          record.prediction ? record.prediction.toUpperCase() : 'N/A';

      card.innerHTML = `
        <div class="logbook-header">
          <h4>${fullName}</h4>
          <span>${timestamp}</span>
        </div>
        <div class="logbook-grid">
          <p><strong>Patient ID:</strong> ${details.patient_id || 'N/A'}</p>
          <p><strong>Age/Gender:</strong> ${details.age || 'N/A'} / ${
          details.gender || 'N/A'}</p>
          <p><strong>Eye:</strong> ${details.eye_examined || 'N/A'}</p>
          <p><strong>Allergies:</strong> ${allergies}</p>
          <p><strong>Diagnosis:</strong> ${diagnosis}</p>
          <p><strong>Mode:</strong> ${
          record.mode || record.record_type || 'patient-only'}</p>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (error) {
    container.innerHTML = `<p style="color:#ff8f8f;">${error.message}</p>`;
  }
}

// ========== SERVER HEALTH CHECK ==========

async function checkServerHealth() {
  try {
    const response = await fetchWithAuth('/api/health');
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
    const statusText = document.getElementById('status-text');
    if (statusText) {
      statusText.textContent = 'Error: Cannot connect to server';
    }
  }
}

// ========== UTILITIES ==========

function playNotificationSound() {
  try {
    const audio = document.getElementById('notification-sound');
    if (audio) {
      audio.play().catch(
          () => {
              // Silently fail if audio cannot play
          });
    }
  } catch (e) {
    // Silently ignore audio errors
  }
}

// ========== DIAGNOSTIC MODAL FUNCTIONS ==========

function openDiagnosticModal() {
  const result = globalState.currentAnalysis;
  if (!result || !result.clinical_metrics) {
    console.log('No clinical metrics available');
    return;
  }

  const metrics = result.clinical_metrics;
  const quality = result.segmentation_quality || {};
  const metricsValid = metrics.is_valid !== false;

  // Populate modal with metrics
  document.getElementById('modal-glaucoma-pct').textContent =
      metrics.glaucoma_percentage + '%';
  document.getElementById('modal-sensitivity').textContent =
      metricsValid ? (metrics.sensitivity + '%') : 'N/A';
  document.getElementById('modal-specificity').textContent =
      metricsValid ? (metrics.specificity + '%') : 'N/A';
  document.getElementById('modal-cdr').textContent =
      (metricsValid && typeof metrics.cup_to_disc_ratio === 'number') ?
      metrics.cup_to_disc_ratio.toFixed(3) :
      'N/A';

  // Set severity with color coding
  const severityEl = document.getElementById('modal-severity');
  severityEl.textContent = metrics.severity;
  if (metricsValid) {
    severityEl.className =
        'severity-value ' + metrics.severity.toLowerCase().replace(/\s+/g, '-');
  } else {
    severityEl.className = 'severity-value unreliable';
  }

  // Set recommendation
  const recommendationEl = document.getElementById('modal-recommendation');
  recommendationEl.textContent = metrics.recommendation;
  if (!metricsValid && quality.reason) {
    recommendationEl.textContent += ` (${quality.reason})`;
  }

  // Show modal
  document.getElementById('diagnostic-modal').style.display = 'flex';
}

function closeDiagnosticModal() {
  document.getElementById('diagnostic-modal').style.display = 'none';
}

// Close modal when clicking outside of it
window.addEventListener('click', (event) => {
  const modal = document.getElementById('diagnostic-modal');
  const settingsModal = document.getElementById('settings-modal');
  if (event.target === modal) {
    closeDiagnosticModal();
  }
  if (event.target === settingsModal) {
    settingsModal.style.display = 'none';
  }
});
