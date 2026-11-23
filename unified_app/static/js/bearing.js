/**
 * =============================================================================
 * LOGIKA MODUL INSPEKSI CORAN / BEARING (bearing.js)
 * =============================================================================
 * Pemilik: Tim Inspeksi Cacat Coran Logam & Bearing
 * Model: Sequential CNN Pure-NumPy (Grayscale 64x64, tanpa framework deep learning eksternal)
 *        + Lokalisasi Grad-CAM Weakly-Supervised & Matriks Anomali Permukaan 128x128
 * =============================================================================
 */

// Status Modul Coran / Bearing (Modul Sekunder: Dropzone Berkas & Presets Sampel)
let currentCastSens = 'standard'; // 'standard' | 'high' | 'ultra'
let castActiveFilter = 'grad_cam'; // 'grad_cam' | 'anomaly_matrix'
let castLatestResult = null;

// Safe compatibility stubs
function setCastInputMode(mode) {}
function captureCastFrame() {}
function toggleCastAuto() {}
function runCastLoop() {}

/**
 * Pemilih kebijakan sensitivitas inspeksi QC (Standar / Tinggi / Ultra)
 */
function setCastSensitivity(sens) {
  currentCastSens = sens;
  ['sensStd', 'sensHigh', 'sensUltra'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });

  if (sens === 'standard') document.getElementById('sensStd').classList.add('active');
  if (sens === 'high') document.getElementById('sensHigh').classList.add('active');
  if (sens === 'ultra') document.getElementById('sensUltra').classList.add('active');

  // Jalankan ulang inspeksi jika sudah ada citra yang dimuat
  if (castLatestResult && castLatestResult.original_image) {
    submitModuleInference('bearing', { image_base64: castLatestResult.original_image, sensitivity: currentCastSens }, true);
  }
}

/**
 * Pemilih tampilan filter (Grad-CAM vs Matriks Anomali Permukaan)
 */
function selectCastFilter(key) {
  castActiveFilter = key;
  document.getElementById('btnCastCam').classList.toggle('active', key === 'grad_cam');
  document.getElementById('btnCastMatrix').classList.toggle('active', key === 'anomaly_matrix');

  if (key === 'grad_cam') {
    document.getElementById('cOutputTitle').innerText = 'Lokalisasi Cacat Grad-CAM';
    document.getElementById('cDescTitle').innerText = 'Lokalisasi Cacat Weakly-Supervised';
    document.getElementById('cDescText').innerText =
      'Grad-CAM menghitung gradien aktivasi dari lapisan konvolusi terakhir untuk memetakan titik anomali cacat mikro di seluruh permukaan coran logam.';
  } else {
    document.getElementById('cOutputTitle').innerText = 'Matriks Anomali Permukaan 128x128';
    document.getElementById('cDescTitle').innerText = 'Matriks Anomali Permukaan (Heatmap)';
    document.getElementById('cDescText').innerText =
      'Matriks anomali menunjukkan sebaran densitas spasial 2D dari fitur cacat di bidang coran logam berdimensi 128x128 sel.';
  }

  if (castLatestResult) updateCastFrameViews(castLatestResult);
}

/**
 * Penanganan unggah berkas citra
 */
function handleCastFile(e) {
  if (e.target.files.length > 0) {
    const fd = new FormData();
    fd.append('file', e.target.files[0]);
    fd.append('sensitivity', currentCastSens);
    submitModuleInference('bearing', fd, false);
  }
}

/**
 * Muat contoh preset coran (Cacat atau Normal)
 */
function loadCastSample(sampleType) {
  const ph1 = document.getElementById('cPh1');
  const namaSampel = sampleType === 'defect' ? 'cacat' : 'normal';
  if (ph1) {
    ph1.innerText = `Mengambil sampel coran ${namaSampel}...`;
    ph1.style.display = 'block';
  }
  const img1 = document.getElementById('cImg1');
  if (img1) img1.style.display = 'none';

  const tryFetch = (url, fallbackUrl) => {
    return fetch(url).then(res => {
      if (res.ok) return res.blob();
      if (fallbackUrl) return tryFetch(fallbackUrl, null);
      throw new Error(`Sampel tidak ditemukan pada server (HTTP ${res.status}). Pastikan server dijalankan dengan: python run.py`);
    });
  };

  const t = Date.now();
  tryFetch(`/api/sample/bearing/${sampleType}?t=${t}`, `/api/sample/${sampleType}?t=${t}`)
    .then(blob => {
      const fd = new FormData();
      fd.append('file', blob, `sample_casting_${sampleType}_${t}.jpg`);
      fd.append('sensitivity', currentCastSens);
      submitModuleInference('bearing', fd, false);
    })
    .catch(err => {
      alert('Gagal memuat sampel coran logam: ' + err.message);
      if (ph1) ph1.innerText = 'Menunggu citra coran logam...';
    });
}

/**
 * Render hasil inspeksi coran logam ke UI
 */
function renderCastResults(data) {
  castLatestResult = data;
  const isDefect = data.is_defect;
  const isReview = data.predicted_label === 'Perlu Review' || data.predicted_label === 'Review Required';

  // Indikator latensi (aman jika elemen badge tidak ada di DOM)
  const cLatBadge = document.getElementById('cLatencyBadge');
  if (cLatBadge) cLatBadge.innerText = `${data.processing_ms} ms`;
  const cKpiLat = document.getElementById('cKpiLatency');
  if (cKpiLat) cKpiLat.innerText = `${data.processing_ms} ms`;
  setImg('cImg1', 'cPh1', data.original_image);
  updateCastFrameViews(data);

  const cardOut = document.getElementById('cCardOutput');
  const tagOut = document.getElementById('cOutputTag');
  if (isDefect) {
    cardOut.className = 'display-viewport-card glow-defect';
    tagOut.className = 'frame-badge-tag badge-defect';
    tagOut.innerText = 'CACAT (DEFECT)';
  } else if (isReview) {
    cardOut.className = 'display-viewport-card';
    cardOut.style.borderColor = '#F59E0B';
    tagOut.className = 'frame-badge-tag';
    tagOut.style.background = '#FFFBEB';
    tagOut.style.color = '#D97706';
    tagOut.innerText = 'REVIEW';
  } else {
    cardOut.className = 'display-viewport-card glow-pass';
    tagOut.className = 'frame-badge-tag badge-output';
    tagOut.innerText = 'NORMAL (OK)';
  }

  // Banner Keputusan
  const banner = document.getElementById('cDecisionBanner');
  banner.className = 'decision-banner-box ' + (isDefect ? 'reject' : (isReview ? 'review' : 'pass'));
  
  const displayDecision = isDefect ? 'REJECT (DEFECT)' : (isReview ? 'PERLU REVIEW OPERATOR' : 'PASS (OK)');
  document.getElementById('cDecisionTxt').innerText = displayDecision;
  document.getElementById('cConfTxt').innerText = `Keyakinan: ${data.confidence}% (Sequential CNN Pure-NumPy)`;

  // Ikon Keputusan Dinamis
  const cIcon = document.getElementById('cDecisionIcon');
  if (cIcon) {
    if (isDefect) {
      cIcon.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
    } else if (isReview) {
      cIcon.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>`;
    } else {
      cIcon.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
    }
  }

  // Metrik Kualitas Citra
  const q = data.quality_details || {};
  document.getElementById('castBlurVal').innerText = q.blur_score ? `${q.blur_score}` : '-';
  document.getElementById('castBrightVal').innerText = q.mean_brightness ? `${q.mean_brightness}` : '-';
  document.getElementById('castUncertVal').innerText = isReview ? 'Ya (Perlu Review)' : 'Normal (Aman)';

  // Bar Distribusi Probabilitas
  const probs = data.probabilities || {};
  const defP = probs.Defective || 0;
  const okP = probs.OK || 0;
  document.getElementById('cBarDefect').style.width = `${defP}%`;
  document.getElementById('cValDefect').innerText = `${defP}%`;
  document.getElementById('cBarOk').style.width = `${okP}%`;
  document.getElementById('cValOk').innerText = `${okP}%`;
}

/**
 * Perbarui tampilan viewport output sesuai filter aktif
 */
function updateCastFrameViews(data) {
  if (castActiveFilter === 'anomaly_matrix') {
    setImg('cImg2', 'cPh2', data.anomaly_matrix || data.annotated_image);
  } else {
    setImg('cImg2', 'cPh2', data.annotated_image || data.original_image);
  }
}
