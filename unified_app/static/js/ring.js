/**
 * =============================================================================
 * LOGIKA MODUL INSPEKSI RING BEARING (ring.js)
 * =============================================================================
 * Pemilik: Tim Inspeksi Ring Bearing
 * Model: Pure 7-Stage Deep CNN (PyTorch, 512x512 CLAHE Grayscale) + Metrology Safety Interlock
 * =============================================================================
 */

// Status Modul Ring Bearing
let isRingAuto = false;
let ringAutoTimer = null;
let isRingProcessing = false;
let ringActiveFilter = 'red_dots';
let ringMatrixDim = 512;
let ringScanInterval = 800;
let ringLatestResult = null;

// Opsi resolusi diskrit untuk slider metrologi (128x128 hingga 4096x4096)
const MATRIX_RESOLUTIONS = [128, 256, 512, 1024, 2048, 4096];

// Deskripsi Teknis Tampilan Filter Visual
const RING_FILTER_DESCRIPTIONS = {
  'red_dots': {
    title: 'Lokalisasi Cacat (Kotak Pembatas pada Citra Asli)',
    tag: 'Standar Metrologi QC',
    sub: 'Citra Tajam Tanpa Blur • Deteksi Klaster Anomali Deterministik',
    text: 'Kotak pembatas merah diekstraksi langsung dari klaster anomali berdensitas tinggi pada matriks metrologi disparitas tanpa blur artifisial, mempertahankan detail tekstur asli komponen.'
  },
  'chessboard_matrix': {
    title: 'Matriks Metrologi Disparitas',
    tag: 'Analisis Numerik Dinamis',
    sub: 'Kisi Heatmap Metrologi Dinamis (128x128 hingga 4096x4096 sel)',
    text: 'Visualisasi kuantitatif disparitas residual piksel permukaan terhadap baseline ring normal. Resolusi matriks dapat diatur fleksibel hingga 4096x4096 sel.'
  },
  'filter_clahe': {
    title: 'Filter CLAHE (Contrast Limited Adaptive Histogram Equalization)',
    tag: 'Standar Permukaan Industri',
    sub: 'Normalisasi Pencahayaan Adaptif & Eliminasi Glare Kilau Logam',
    text: 'Menyamakan intensitas pencahayaan di seluruh lengkungan kelengkungan logam, membedakan pantulan cahaya kilau dengan goresan mikro cacat secara akurat.'
  }
};

/**
 * Beralih mode input antara Unggah Berkas dan Kamera Langsung
 */
function setRingInputMode(mode) {
  document.getElementById('ringTabUpload').classList.toggle('active', mode === 'upload');
  document.getElementById('ringTabCam').classList.toggle('active', mode === 'camera');
  document.getElementById('ringUploadArea').style.display = mode === 'upload' ? 'block' : 'none';
  document.getElementById('ringCameraArea').style.display = mode === 'camera' ? 'block' : 'none';

  if (mode === 'camera') {
    startCamera('ringVideo');
  } else {
    stopCamera();
  }
}

/**
 * Pengaturan interval pemindaian kamera otomatis
 */
function setScanRate(val) {
  ringScanInterval = parseInt(val, 10) || 800;
}

/**
 * Penanganan Slider Resolusi Matriks Metrologi
 */
function onMatrixSliderInput(sliderIdx) {
  const idx = parseInt(sliderIdx, 10) || 0;
  const dim = MATRIX_RESOLUTIONS[idx] || 512;
  ringMatrixDim = dim;
  const totalCells = (dim * dim).toLocaleString('id-ID');
  document.getElementById('activeResLabel').innerText = `${dim}x${dim} aktif (${totalCells} sel)`;
  updateSliderPointsUI(idx);
}

function onMatrixSliderChange(sliderIdx) {
  const idx = parseInt(sliderIdx, 10) || 0;
  const dim = MATRIX_RESOLUTIONS[idx] || 512;
  ringMatrixDim = dim;
  const totalCells = (dim * dim).toLocaleString('id-ID');
  document.getElementById('activeResLabel').innerText = `${dim}x${dim} aktif (${totalCells} sel)`;
  updateSliderPointsUI(idx);

  // Otomatis beralih ke tampilan matriks agar operator langsung melihat grid
  selectRingFilter('chessboard_matrix');

  if (ringLatestResult && ringLatestResult.original_image) {
    submitModuleInference('ring', { image_base64: ringLatestResult.original_image, grid_dim: dim }, true);
  }
}

function setSliderIndex(idx) {
  const slider = document.getElementById('ringMatrixSlider');
  if (slider) {
    slider.value = idx;
    onMatrixSliderInput(idx);
    onMatrixSliderChange(idx);
  }
}

function updateSliderPointsUI(activeIdx) {
  document.querySelectorAll('.slider-tick-item').forEach((item, i) => {
    item.classList.toggle('active', i === parseInt(activeIdx, 10));
  });
}

/**
 * Pilih tampilan visual filter di viewport kanan
 */
function selectRingFilter(filterKey) {
  ringActiveFilter = filterKey;
  ['btnFilterDefect', 'btnFilterMatrix', 'btnFilterClahe'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.classList.remove('active');
  });

  if (filterKey === 'red_dots') {
    const b = document.getElementById('btnFilterDefect');
    if (b) b.classList.add('active');
  } else if (filterKey === 'chessboard_matrix') {
    const b = document.getElementById('btnFilterMatrix');
    if (b) b.classList.add('active');
  } else if (filterKey === 'filter_clahe') {
    const b = document.getElementById('btnFilterClahe');
    if (b) b.classList.add('active');
  }

  const desc = RING_FILTER_DESCRIPTIONS[filterKey] || RING_FILTER_DESCRIPTIONS['red_dots'];
  document.getElementById('rDescTitle').innerText = desc.title;
  document.getElementById('rDescTag').innerText = desc.tag;
  document.getElementById('rDescSub').innerText = desc.sub;
  document.getElementById('rDescText').innerText = desc.text;
  document.getElementById('rOutputTitle').innerText = desc.title;

  if (ringLatestResult) updateRingFrameViews(ringLatestResult);
}

/**
 * Penanganan unggah berkas
 */
function handleRingFile(e) {
  if (e.target.files.length > 0) {
    const fd = new FormData();
    fd.append('file', e.target.files[0]);
    fd.append('grid_dim', ringMatrixDim);
    submitModuleInference('ring', fd, false);
  }
}

/**
 * Penanganan tombol snapshot kamera
 */
function captureRingFrame() {
  const dataUrl = captureCanvasFrame('ringVideo');
  if (dataUrl) {
    submitModuleInference('ring', { image_base64: dataUrl, grid_dim: ringMatrixDim }, true);
  }
}

/**
 * Toggle loop inspeksi real-time berkelanjutan
 */
function toggleRingAuto() {
  if (isRingAuto) {
    isRingAuto = false;
    if (ringAutoTimer) clearTimeout(ringAutoTimer);
    document.getElementById('btnRingAuto').classList.remove('running');
    document.getElementById('ringAutoTxt').innerText = 'Mulai Real-Time';
  } else {
    isRingAuto = true;
    document.getElementById('btnRingAuto').classList.add('running');
    document.getElementById('ringAutoTxt').innerText = 'Hentikan Real-Time';
    runRingLoop();
  }
}

function runRingLoop() {
  if (!isRingAuto) return;
  if (isRingProcessing) {
    ringAutoTimer = setTimeout(runRingLoop, 50);
    return;
  }

  const dataUrl = captureCanvasFrame('ringVideo');
  if (!dataUrl) {
    ringAutoTimer = setTimeout(runRingLoop, 150);
    return;
  }

  isRingProcessing = true;
  fetch('/api/v1/inspect/ring', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_base64: dataUrl, grid_dim: ringMatrixDim })
  })
    .then(res => res.json())
    .then(data => {
      isRingProcessing = false;
      if (data.status === 'success') renderRingResults(data);
      if (isRingAuto) ringAutoTimer = setTimeout(runRingLoop, ringScanInterval);
    })
    .catch(() => {
      isRingProcessing = false;
      if (isRingAuto) ringAutoTimer = setTimeout(runRingLoop, 1000);
    });
}

/**
 * Muat contoh preset ring (Cacat atau Normal)
 */
function loadRingSample(sampleType) {
  const ph1 = document.getElementById('rPh1');
  const namaSampel = sampleType === 'defect' ? 'cacat' : 'normal';
  if (ph1) {
    ph1.innerText = `Mengambil sampel ring ${namaSampel}...`;
    ph1.style.display = 'block';
  }
  const img1 = document.getElementById('rImg1');
  if (img1) img1.style.display = 'none';

  // Coba route spesifik modul ring terlebih dahulu, dengan fallback ke route umum
  const tryFetch = (url, fallbackUrl) => {
    return fetch(url).then(res => {
      if (res.ok) return res.blob();
      if (fallbackUrl) return tryFetch(fallbackUrl, null);
      throw new Error(`Sampel tidak ditemukan pada server (HTTP ${res.status}). Pastikan server dijalankan dengan: python run.py`);
    });
  };

  const t = Date.now();
  tryFetch(`/api/sample/ring/${sampleType}?t=${t}`, `/api/sample/${sampleType}?t=${t}`)
    .then(blob => {
      const fd = new FormData();
      fd.append('file', blob, `sample_${sampleType}_${t}.jpg`);
      fd.append('grid_dim', ringMatrixDim);
      submitModuleInference('ring', fd, false);
    })
    .catch(err => {
      alert('Gagal memuat sampel ring bearing: ' + err.message);
      if (ph1) ph1.innerText = 'Menunggu citra ring bearing...';
    });
}

/**
 * Render hasil lengkap inspeksi ring bearing ke UI
 */
function renderRingResults(data) {
  ringLatestResult = data;
  const isDefect = data.is_defect;

  // Indikator latensi
  document.getElementById('rLatencyBadge').innerText = `${data.processing_ms} ms`;
  const kpiLat = document.getElementById('rKpiLatency');
  if (kpiLat) kpiLat.innerText = `${data.processing_ms} ms`;

  setImg('rImg1', 'rPh1', data.original_image);
  updateRingFrameViews(data);

  const cardOut = document.getElementById('rCardOutput');
  const tagOut = document.getElementById('rOutputTag');
  if (isDefect) {
    cardOut.className = 'display-viewport-card glow-defect';
    tagOut.className = 'frame-badge-tag badge-defect';
    tagOut.innerText = `${data.num_defect_markers || 'ANOMALI'} CACAT`;
  } else {
    cardOut.className = 'display-viewport-card glow-pass';
    tagOut.className = 'frame-badge-tag badge-output';
    tagOut.innerText = 'PASS (NORMAL)';
  }

  // Banner Keputusan
  const banner = document.getElementById('rDecisionBanner');
  banner.className = 'decision-banner-box ' + (isDefect ? 'reject' : 'pass');
  document.getElementById('rDecisionTxt').innerText = `${data.decision}`;

  const gateLabel = data.gate_info ? ` [${data.gate_info}]` : ' (Pure 7-Stage Deep CNN)';
  document.getElementById('rConfTxt').innerText = `Keyakinan: ${data.confidence}%${gateLabel}`;

  // Badge Gate
  const gateBadge = document.getElementById('rGateBadge');
  if (gateBadge) {
    gateBadge.innerText = data.gate_info || (isDefect ? 'Cacat' : 'Normal');
    if (isDefect) {
      gateBadge.style.background = '#FEF2F2';
      gateBadge.style.color = '#DC2626';
      gateBadge.style.borderColor = '#FECACA';
    } else {
      gateBadge.style.background = '#ECFDF5';
      gateBadge.style.color = '#059669';
      gateBadge.style.borderColor = '#A7F3D0';
    }
  }

  // Ikon Keputusan Dinamis
  const iconWrap = document.getElementById('rDecisionIcon');
  if (iconWrap) {
    if (isDefect) {
      iconWrap.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
    } else {
      iconWrap.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
    }
  }

  // Bar Probabilitas
  const probs = data.probabilities || {};
  const defP = probs.defect_ring || 0;
  const normP = probs.normal_ring || 0;
  document.getElementById('rBarDefect').style.width = `${defP}%`;
  document.getElementById('rValDefect').innerText = `${defP}%`;
  document.getElementById('rBarNormal').style.width = `${normP}%`;
  document.getElementById('rValNormal').innerText = `${normP}%`;

  // KPI Metrologi Kuantitatif
  const metrics = data.metrics || {};
  const kpiDef = document.getElementById('rKpiDefects');
  if (kpiDef) kpiDef.innerHTML = `${data.num_defect_markers || 0} <small>klaster</small>`;

  const kpiMat = document.getElementById('rKpiMatrix');
  if (kpiMat) kpiMat.innerText = metrics.matrix_shape || `${ringMatrixDim}x${ringMatrixDim}`;

  const kpiSobel = document.getElementById('rKpiSobel');
  if (kpiSobel) {
    kpiSobel.innerHTML =
      metrics.sobel_surface_gradient !== undefined
        ? `${metrics.sobel_surface_gradient} <small>(Ambang: 26.0)</small>`
        : '-';
  }
}

/**
 * Perbarui tampilan viewport output sesuai filter aktif
 */
function updateRingFrameViews(data) {
  const views = data.views || {};
  let targetImg = data.annotated_image || views.red_dots;

  if (ringActiveFilter === 'chessboard_matrix') targetImg = views.chessboard_matrix || targetImg;
  else if (ringActiveFilter === 'filter_clahe') targetImg = views.filter_clahe || targetImg;

  setImg('rImg2', 'rPh2', targetImg);
}
  