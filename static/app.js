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
    title: 'Input',
    description:
        'Upload a retinal fundus image. Better image quality produces more stable predictions and cleaner segmentation masks.'
  },
  preprocess: {
    title: 'Preprocess',
    description:
        'The image is normalized and resized to the model input size so downstream inference remains consistent.'
  },
  classify: {
    title: 'Classify (EfficientNet-B0)',
    description:
        'EfficientNet-B0 estimates glaucoma probability and confidence from global retinal patterns.'
  },
  segment: {
    title: 'Segment (DeepLabV3+)',
    description:
        'DeepLabV3+ extracts optic disc and cup regions, giving visual evidence for structural interpretation.'
  },
  report: {
    title: 'Final Report',
    description:
        'Prediction scores and anatomical overlays are combined into a readable clinical decision-support view.'
  }
};

const workplacePresentationSlides = [
  {
    title: 'Project Vision and Clinical Need',
    subtitle: 'Why this system matters in real-world screening',
    visual: 'hero',
    bullets: [
      'Glaucoma is often silent in early stages, making proactive screening critical.',
      'The project delivers AI-assisted triage to help identify high-risk cases faster.',
      'Combines predictive confidence and anatomical evidence for better clinician trust.',
      'Designed for practical workflow: upload, inference, explainability, and documentation.'
    ]
  },
  {
    title: 'Problem Statement and Objectives',
    subtitle: 'Target outcomes from engineering and clinical perspectives',
    visual: 'barChart',
    bullets: [
      'Primary objective: classify glaucoma likelihood from retinal fundus images.',
      'Secondary objective: segment optic disc and cup to support structural interpretation.',
      'Operational objective: provide fast response time for clinic-friendly usage.',
      'Documentation objective: retain patient-associated analysis records in the logbook.'
    ]
  },
  {
    title: 'End-to-End Workplace Flow',
    subtitle: 'How data moves from upload to clinical report',
    visual: 'pipeline',
    bullets: [
      'Step 1: intake and validation of image files with secure handling.',
      'Step 2: preprocessing transforms and normalization for model consistency.',
      'Step 3: EfficientNet-B0 classification for risk probability estimation.',
      'Step 4: DeepLabV3+ segmentation and post-processing refinements.',
      'Step 5: final report with confidence, CDR signal, overlays, and recommendation.'
    ]
  },
  {
    title: 'Classification Model Deep Dive',
    subtitle: 'EfficientNet-B0 in the diagnosis path',
    visual: 'modelBlocks',
    bullets: [
      'EfficientNet-B0 backbone balances accuracy and computational efficiency.',
      'Fine-tuned for binary glaucoma risk prediction using retinal datasets.',
      'Outputs calibrated class probabilities for transparent clinical messaging.',
      'Supports single and batch pathways in the same API architecture.'
    ]
  },
  {
    title: 'Segmentation Model Deep Dive',
    subtitle: 'DeepLabV3+ and structural explainability',
    visual: 'heatmap',
    bullets: [
      'DeepLabV3+ isolates optic disc and cup regions for structural analysis.',
      'Segmentation is refined with morphology and anatomical constraints.',
      'Improves interpretability by showing where the model attends spatially.',
      'Supports cup-to-disc ratio estimation as a clinically meaningful indicator.'
    ]
  },
  {
    title: 'Python Stack and Libraries',
    subtitle: 'Core tooling used across backend and inference',
    visual: 'libraryGrid',
    bullets: [
      'PyTorch and torchvision for loading, transforming, and running deep models.',
      'OpenCV + Pillow for image operations, overlays, and segmentation mask processing.',
      'NumPy for numerical logic, thresholding behavior, and quality calculations.',
      'Flask + Werkzeug for API routing, authentication, and secure uploads.'
    ]
  },
  {
    title: 'Code Architecture Overview',
    subtitle: 'How the application is organized in code',
    visual: 'codeFlow',
    bullets: [
      'Model loading functions initialize classification and segmentation weights.',
      'Dedicated routes expose classify, segment, combined, health, and logbook APIs.',
      'Helper functions handle preprocessing, mask refinement, and quality checks.',
      'Frontend JavaScript orchestrates UI interactions and endpoint communication.'
    ]
  },
  {
    title: 'Segmentation Robustness Strategy',
    subtitle: 'Fallback and refinement logic for difficult images',
    visual: 'lineChart',
    bullets: [
      'Multi-pass inference supports stability when lighting or focus is inconsistent.',
      'Intensity-based fallback recovers disc/cup candidates in weak-mask scenarios.',
      'Component selection near priors limits drift to irrelevant bright regions.',
      'Quality scoring flags low-confidence geometry before reporting.'
    ]
  },
  {
    title: 'Outputs, Metrics, and Interpretability',
    subtitle: 'What the clinician and presenter can explain clearly',
    visual: 'resultsPanel',
    bullets: [
      'Classification risk and confidence summarize disease likelihood quickly.',
      'Disc/cup overlays provide visual rationale beyond a raw probability number.',
      'Cup-to-disc ratio trend helps communicate severity context.',
      'Heatmap-style evidence improves explainability during presentation.'
    ]
  },
  {
    title: 'Patient Workflow and Logbook',
    subtitle: 'From analysis to traceable records',
    visual: 'timeline',
    bullets: [
      'Optional patient details keep analysis fast without forced form completion.',
      'Save-to-logbook captures patient-linked assessment for follow-up.',
      'Logbook tab enables retrieval and review of historical entries.',
      'Supports continuity in screening campaigns and case discussions.'
    ]
  },
  {
    title: 'Deployment Readiness',
    subtitle: 'How the project runs in current form',
    visual: 'deployment',
    bullets: [
      'Flask app supports authenticated usage with structured API responses.',
      'Device strategy uses CUDA/MPS/CPU fallback for portability.',
      'Web interface supports single image analysis and batch processing modes.',
      'Current architecture can be extended into cloud-hosted clinical services.'
    ]
  },
  {
    title: 'Future Improvements and Research Path',
    subtitle: 'Where this system can evolve next',
    visual: 'roadmap',
    bullets: [
      'Broader datasets and external validation to improve generalization.',
      'Model ensembling and calibration for stronger robustness.',
      'Potential EHR integration and physician feedback loops.',
      'Prospective studies for real-world impact on screening efficiency.'
    ]
  }
];

let currentWorkplaceSlideIndex = 0;

// ========== INITIALIZATION ==========

document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  setupSingleAnalysisStepper();
  setupDisplayControls();
  setupWorkplaceFlowchart();
  setupWorkplacePresentationModal();
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
}

function setupSingleAnalysisStepper() {
  const patientPanel = document.getElementById('step-panel-patient');
  const uploadPanel = document.getElementById('step-panel-upload');
  const nextBtn = document.getElementById('go-to-upload-btn');
  const backBtn = document.getElementById('back-to-patient-btn');

  if (!patientPanel || !uploadPanel) {
    return;
  }

  const setStep = (step) => {
    const showUpload = step === 'upload';
    patientPanel.classList.toggle('active', !showUpload);
    patientPanel.setAttribute('aria-hidden', showUpload ? 'true' : 'false');
    uploadPanel.classList.toggle('active', showUpload);
    uploadPanel.setAttribute('aria-hidden', showUpload ? 'false' : 'true');
  };

  nextBtn?.addEventListener('click', () => setStep('upload'));
  backBtn?.addEventListener('click', () => setStep('patient'));

  setStep('patient');
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

  if (!nodes.length) {
    return;
  }

  const setActiveNode = (selectedNode) => {
    nodes.forEach(node => node.classList.remove('active'));
    selectedNode.classList.add('active');
  };

  nodes.forEach(node => {
    node.addEventListener('click', () => setActiveNode(node));
  });

  setActiveNode(nodes[0]);
}

function setupWorkplacePresentationModal() {
  const openBtn = document.getElementById('open-ppt-modal-btn');
  const closeBtn = document.getElementById('close-ppt-modal-btn');
  const modal = document.getElementById('workplace-ppt-modal');
  const prevBtn = document.getElementById('ppt-prev-btn');
  const nextBtn = document.getElementById('ppt-next-btn');
  const dotsContainer = document.getElementById('ppt-dots');

  if (!openBtn || !closeBtn || !modal || !prevBtn || !nextBtn ||
      !dotsContainer) {
    return;
  }

  dotsContainer.innerHTML = '';
  workplacePresentationSlides.forEach((_, index) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'ppt-dot';
    dot.setAttribute('aria-label', `Go to slide ${index + 1}`);
    dot.addEventListener('click', () => goToPresentationSlide(index));
    dotsContainer.appendChild(dot);
  });

  openBtn.addEventListener('click', () => {
    modal.style.display = 'flex';
    renderPresentationSlide();
  });

  closeBtn.addEventListener('click', () => {
    modal.style.display = 'none';
  });

  modal.addEventListener('click', (event) => {
    if (event.target === modal) {
      modal.style.display = 'none';
    }
  });

  prevBtn.addEventListener('click', () => changePresentationSlide(-1));
  nextBtn.addEventListener('click', () => changePresentationSlide(1));

  document.addEventListener('keydown', (event) => {
    if (modal.style.display !== 'flex') {
      return;
    }
    if (event.key === 'ArrowRight') {
      changePresentationSlide(1);
    } else if (event.key === 'ArrowLeft') {
      changePresentationSlide(-1);
    } else if (event.key === 'Escape') {
      modal.style.display = 'none';
    }
  });
}

function changePresentationSlide(offset) {
  const total = workplacePresentationSlides.length;
  currentWorkplaceSlideIndex =
      (currentWorkplaceSlideIndex + offset + total) % total;
  renderPresentationSlide();
}

function goToPresentationSlide(index) {
  currentWorkplaceSlideIndex = index;
  renderPresentationSlide();
}

function renderPresentationSlide() {
  const titleEl = document.getElementById('ppt-slide-title');
  const subtitleEl = document.getElementById('ppt-subtitle');
  const bulletsEl = document.getElementById('ppt-slide-bullets');
  const counterEl = document.getElementById('ppt-counter');
  const visualEl = document.getElementById('ppt-visual-panel');
  const dots = document.querySelectorAll('#ppt-dots .ppt-dot');

  if (!titleEl || !subtitleEl || !bulletsEl || !counterEl || !visualEl) {
    return;
  }

  const slide = workplacePresentationSlides[currentWorkplaceSlideIndex];
  titleEl.textContent = slide.title;
  subtitleEl.textContent = slide.subtitle || '';
  counterEl.textContent = `Slide ${currentWorkplaceSlideIndex + 1} / ${
      workplacePresentationSlides.length}`;

  bulletsEl.innerHTML = '';
  slide.bullets.forEach((bullet) => {
    const li = document.createElement('li');
    li.textContent = bullet;
    bulletsEl.appendChild(li);
  });

  visualEl.innerHTML = buildSlideVisual(slide.visual);

  dots.forEach((dot, index) => {
    dot.classList.toggle('active', index === currentWorkplaceSlideIndex);
  });
}

function buildSlideVisual(visualType) {
  const templates = {
    hero: `
      <div class="ppt-hero-image">
        <div class="eye-ring"></div>
        <div class="eye-core"></div>
        <div class="heat-spot"></div>
        <p>Retinal AI screening overview</p>
      </div>
    `,
    barChart: `
      <div class="ppt-chart-wrap">
        <h4>Target Outcome Coverage</h4>
        <div class="ppt-bar-row"><span>Detection Speed</span><i style="width:86%"></i></div>
        <div class="ppt-bar-row"><span>Explainability</span><i style="width:91%"></i></div>
        <div class="ppt-bar-row"><span>Clinical Readability</span><i style="width:88%"></i></div>
        <div class="ppt-bar-row"><span>Workflow Continuity</span><i style="width:84%"></i></div>
      </div>
    `,
    pipeline: `
      <div class="ppt-pipeline">
        <span>Input</span><em>→</em><span>Preprocess</span><em>→</em><span>Classify</span><em>→</em><span>Segment</span><em>→</em><span>Report</span>
      </div>
    `,
    modelBlocks: `
      <div class="ppt-block-grid">
        <article><h4>Stem</h4><p>Early edge and vessel cues</p></article>
        <article><h4>MBConv Stack</h4><p>Efficient multi-scale patterns</p></article>
        <article><h4>Feature Head</h4><p>Global context extraction</p></article>
        <article><h4>Classifier</h4><p>Glaucoma probability score</p></article>
      </div>
    `,
    heatmap: `
      <div class="ppt-heatmap-wrap">
        <div class="ppt-retina-base"></div>
        <div class="ppt-heat-layer"></div>
        <div class="ppt-cup-disc"></div>
      </div>
    `,
    libraryGrid: `
      <div class="ppt-library-grid">
        <span>PyTorch</span><span>torchvision</span><span>OpenCV</span>
        <span>Pillow</span><span>NumPy</span><span>Flask</span>
        <span>Werkzeug</span><span>JSON</span><span>Session APIs</span>
      </div>
    `,
    codeFlow: `
      <div class="ppt-code-flow">
        <div>load_models()</div>
        <div>run_segmentation_pipeline()</div>
        <div>compute_quality()</div>
        <div>/api/combined</div>
        <div>frontend render</div>
      </div>
    `,
    lineChart: `
      <div class="ppt-line-chart">
        <svg viewBox="0 0 420 220" preserveAspectRatio="none">
          <polyline points="20,180 90,130 160,120 230,90 300,80 380,55" />
          <circle cx="20" cy="180" r="4"/><circle cx="90" cy="130" r="4"/><circle cx="160" cy="120" r="4"/><circle cx="230" cy="90" r="4"/><circle cx="300" cy="80" r="4"/><circle cx="380" cy="55" r="4"/>
        </svg>
        <p>Segmentation stability improves after fallback/refinement stages.</p>
      </div>
    `,
    resultsPanel: `
      <div class="ppt-results-grid">
        <article><h4>Risk</h4><p>0.82</p></article>
        <article><h4>CDR</h4><p>0.64</p></article>
        <article><h4>Sensitivity</h4><p>91%</p></article>
        <article><h4>Specificity</h4><p>88%</p></article>
      </div>
    `,
    timeline: `
      <div class="ppt-timeline">
        <span>Patient Details</span>
        <span>Image Analysis</span>
        <span>Result Review</span>
        <span>Logbook Save</span>
        <span>Follow-Up</span>
      </div>
    `,
    deployment: `
      <div class="ppt-deployment-grid">
        <div>Flask App</div><div>GPU/CPU Device Layer</div><div>REST Endpoints</div>
        <div>Session Auth</div><div>Batch Processor</div><div>Patient Logbook</div>
      </div>
    `,
    roadmap: `
      <div class="ppt-roadmap">
        <div><strong>Phase 1</strong><p>Dataset expansion</p></div>
        <div><strong>Phase 2</strong><p>Model ensemble</p></div>
        <div><strong>Phase 3</strong><p>EHR integration</p></div>
        <div><strong>Phase 4</strong><p>Prospective trials</p></div>
      </div>
    `
  };

  return templates[visualType] || templates.hero;
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
  if (tabName === 'single-analysis') {
    document.getElementById('step-panel-patient')?.classList.add('active');
    document.getElementById('step-panel-patient')
        ?.setAttribute('aria-hidden', 'false');
    document.getElementById('step-panel-upload')?.classList.remove('active');
    document.getElementById('step-panel-upload')
        ?.setAttribute('aria-hidden', 'true');
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
  if (!details.first_name || !details.last_name || !details.age ||
      !details.gender) {
    alert(
        'Please complete required fields: first name, last name, age, and gender.');
    return false;
  }
  if (details.allergies === 'yes' && !details.allergy_details) {
    alert('Please mention allergy details when allergies are marked Yes.');
    return false;
  }
  return true;
}

function hasAnyPatientDetails(details) {
  return Object.values(details || {})
      .some(value => String(value || '').trim() !== '');
}

function renderPatientSummary(details) {
  const summaryEl = document.getElementById('patient-summary-text');
  if (!summaryEl) {
    return;
  }

  if (!details || !hasAnyPatientDetails(details)) {
    summaryEl.textContent =
        'No patient details were provided for this analysis.';
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
  const hasDetails = hasAnyPatientDetails(patientDetails);
  globalState.currentPatientDetails = hasDetails ? patientDetails : null;

  document.getElementById('single-loading').classList.remove('hidden');
  document.getElementById('analyze-btn').disabled = true;

  try {
    const formData = new FormData();
    formData.append('image', globalState.currentFile);
    formData.append(
        'patient_details', JSON.stringify(hasDetails ? patientDetails : {}));
    const wantsSave = document.getElementById('save-logbook-checkbox')?.checked;
    const shouldSaveToLogbook = Boolean(wantsSave && hasDetails);
    formData.append('save_to_logbook', shouldSaveToLogbook ? 'true' : 'false');

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
      patient_details: hasDetails ? patientDetails : {}
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
  if (!validatePatientDetails(patientDetails)) {
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
