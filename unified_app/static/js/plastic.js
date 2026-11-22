/**
 * =============================================================================
 * LOGIKA MODUL INSPEKSI CACAT INJEKSI PLASTIK (plastic.js)
 * =============================================================================
 * Pemilik: Tim Inspeksi Cacat Produk Plastik (PlasticDefectionWeb)
 * Model: Feature Pyramid Single-Class CNN (PyTorch 768x768 Scale-Preserved)
 *        + Dekoder Regresi Titik Pusat & Geometri + Non-Maximum Suppression (NMS)
 * =============================================================================
 */

// Status Modul Plastik (Modul Sekunder: Dropzone Berkas & Presets Sampel)
let plasticActiveFilter = 'annotated'; // 'annotated' | 'original'
let plasticLatestResult = null;

/**
 * Pemilih tampilan filter (Hasil Deteksi Bounding Box vs Citra Asli)
 */
function selectPlasticFilter(key) {
  plasticActiveFilter = key;
  const btnDefect = document.getElementById('btnPlasticDefect');
  const btnOrig = document.getElementById('btnPlasticOrig');
  if (btnDefect) btnDefect.classList.toggle('active', key === 'annotated');
  if (btnOrig) btnOrig.classList.toggle('active', key === 'original');

  const titleEl = document.getElementById('pOutputTitle');
  if (key === 'annotated') {
    if (titleEl) titleEl.innerText = 'Deteksi Cacat (Bounding Box pada Citra Asli)';
  } else {
    if (titleEl) titleEl.innerText = 'Tampilan Citra Asli Alami';
  }

  if (plasticLatestResult) updatePlasticFrameViews(plasticLatestResult);
}

/**
 * Penanganan unggah berkas citra plastik
 */
function handlePlasticFile(e) {
  if (e.target.files.length > 0) {
    const fd = new FormData();
    fd.append('file', e.target.files[0]);
    submitModuleInference('plastic', fd, false);
  }
}

/**
 * Muat contoh preset plastik (Cacat atau Normal)
 */
function loadPlasticSample(sampleType) {
  const ph1 = document.getElementById('pPh1');
  const namaSampel = sampleType === 'defect' ? 'cacat' : 'normal';
  if (ph1) {
    ph1.innerText = `Mengambil sampel plastik ${namaSampel}...`;
    ph1.style.display = 'block';
  }
  const img1 = document.getElementById('pImg1');
  if (img1) img1.style.display = 'none';

  const tryFetch = (url, fallbackUrl) => {
    return fetch(url).then(res => {
      if (res.ok) return res.blob();
      if (fallbackUrl) return tryFetch(fallbackUrl, null);
      throw new Error(`Sampel tidak ditemukan pada server (HTTP ${res.status}).`);
    });
  };

  const t = Date.now();
  tryFetch(`/api/sample/plastic/${sampleType}?t=${t}`, `/api/sample/${sampleType}?t=${t}`)
    .then(blob => {
      const fd = new FormData();
      fd.append('file', blob, `sample_plastic_${sampleType}_${t}.jpg`);
      submitModuleInference('plastic', fd, false);
    })
    .catch(err => {
      alert('Gagal memuat sampel produk plastik: ' + err.message);
      if (ph1) ph1.innerText = 'Menunggu citra produk plastik...';
    });
}

/**
 * Render hasil inspeksi produk plastik ke antarmuka pengguna
 */
function renderPlasticResults(data) {
  plasticLatestResult = data;
  const isDefect = data.is_defect;
  const numDefects = data.num_defect_markers || (data.defect_boxes ? data.defect_boxes.length : 0);

  // Badge latensi
  const latBadge = document.getElementById('pLatencyBadge');
  if (latBadge) latBadge.innerText = `${data.processing_ms} ms`;
  const kpiLat = document.getElementById('pKpiLatency');
  if (kpiLat) kpiLat.innerText = `${data.processing_ms} ms`;

  // Tampilkan citra asli
  setImg('pImg1', 'pPh1', data.original_image);
  updatePlasticFrameViews(data);

  // Viewport Frame 2 Glow & Tag
  const cardOut = document.getElementById('pCardOutput');
  const tagOut = document.getElementById('pOutputTag');
  if (cardOut && tagOut) {
    if (isDefect) {
      cardOut.className = 'display-viewport-card glow-defect';
      tagOut.className = 'frame-badge-tag badge-defect';
      tagOut.innerText = 'CACAT (DEFECT)';
    } else {
      cardOut.className = 'display-viewport-card glow-pass';
      tagOut.className = 'frame-badge-tag badge-output';
      tagOut.innerText = 'NORMAL (OK)';
    }
  }

  // Banner Keputusan Akhir QC
  const banner = document.getElementById('pDecisionBanner');
  if (banner) {
    banner.className = 'decision-banner-box ' + (isDefect ? 'reject' : 'pass');
  }

  const gateBadge = document.getElementById('pGateBadge');
  if (gateBadge) {
    gateBadge.innerText = isDefect ? 'Tolak (QC Reject)' : 'Lulus (QC Pass)';
  }

  const decTxt = document.getElementById('pDecisionTxt');
  if (decTxt) {
    decTxt.innerText = isDefect ? 'REJECT (DEFECT TERDETEKSI)' : 'PASS (TIDAK ADA CACAT)';
  }

  const confTxt = document.getElementById('pConfTxt');
  if (confTxt) {
    confTxt.innerText = isDefect
      ? `Terdeteksi ${numDefects} titik cacat · Keyakinan Puncak: ${data.confidence}% (PyTorch FPN CNN)`
      : `Permukaan Normal Bebas Cacat · Keyakinan: ${data.confidence}% (PyTorch FPN CNN)`;
  }

  // Ikon Keputusan Dinamis (Ceklis Hijau vs Silang Merah)
  const pIcon = document.getElementById('pDecisionIcon');
  if (pIcon) {
    if (isDefect) {
      pIcon.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>`;
    } else {
      pIcon.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>`;
    }
  }

  // KPI Mini Grid
  const kpiDef = document.getElementById('pKpiDefects');
  if (kpiDef) kpiDef.innerHTML = `${numDefects} <small>${numDefects === 1 ? 'area' : 'area'}</small>`;

  const kpiTh = document.getElementById('pKpiThreshold');
  if (kpiTh && data.detection_threshold) kpiTh.innerText = data.detection_threshold;

  const kpiDim = document.getElementById('pKpiDim');
  if (kpiDim && data.image_size) {
    kpiDim.innerText = `${data.image_size.width}x${data.image_size.height}`;
  }

  // Batang Distribusi Probabilitas
  const probs = data.probabilities || {};
  const defP = probs.Defective || 0;
  const okP = probs.OK || 0;
  const barDef = document.getElementById('pBarDefect');
  const valDef = document.getElementById('pValDefect');
  const barNorm = document.getElementById('pBarNormal');
  const valNorm = document.getElementById('pValNormal');

  if (barDef) barDef.style.width = `${defP}%`;
  if (valDef) valDef.innerText = `${defP}%`;
  if (barNorm) barNorm.style.width = `${okP}%`;
  if (valNorm) valNorm.innerText = `${okP}%`;

  // Render Tabel Koordinat Cacat
  const tblCard = document.getElementById('pDefectTableCard');
  const tblBody = document.getElementById('pDefectTableBody');
  const tagCount = document.getElementById('pDefectCountTag');

  if (tblCard && tblBody) {
    if (isDefect && data.defect_boxes && data.defect_boxes.length > 0) {
      tblCard.style.display = 'block';
      if (tagCount) tagCount.innerText = `${data.defect_boxes.length} Area Cacat Teridentifikasi`;
      
      let rowsHtml = '';
      data.defect_boxes.forEach(box => {
        const coordsStr = box.box_int ? box.box_int.join(', ') : (box.xyxy ? box.xyxy.join(', ') : '-');
        rowsHtml += `
          <tr>
            <td><span class="plastic-defect-id-badge">#${box.id}</span></td>
            <td><strong>${box.label || 'Cacat Injeksi'}</strong></td>
            <td><span style="color: #DC2626; font-weight: 700;">${box.confidence}%</span></td>
            <td><span class="plastic-coord-code">[${coordsStr}]</span></td>
          </tr>
        `;
      });
      tblBody.innerHTML = rowsHtml;
    } else {
      tblCard.style.display = 'none';
      tblBody.innerHTML = '';
    }
  }
}

/**
 * Perbarui tampilan frame gambar output plastik
 */
function updatePlasticFrameViews(data) {
  if (plasticActiveFilter === 'original') {
    setImg('pImg2', 'pPh2', data.original_image);
  } else {
    setImg('pImg2', 'pPh2', data.annotated_image || data.original_image);
  }
}
