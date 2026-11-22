/**
 * =============================================================================
 * GLOBAL APP CONTROLLER (app.js)
 * =============================================================================
 * Digunakan bersama untuk ketiga modul (Ring, Bearing, Plastic).
 * Memuat:
 *   - Status global aplikasi (activeModule, video streams)
 *   - Navigasi modul & manajemen tampilan
 *   - Kontrol hardware kamera & enumerasi perangkat
 *   - Universal API request dispatcher (submitModuleInference)
 *   - Modal Lightbox Zoom
 *   - Modal Evaluasi Akademik (S2)
 * =============================================================================
 */

// -----------------------------------------------------------------------------
// Status Global Aplikasi
// -----------------------------------------------------------------------------
let activeModule = 'ring'; // 'ring' | 'bearing' | 'plastic'
let activeVideoStream = null;

// -----------------------------------------------------------------------------
// Pergantian Modul (Tab Navbar)
// -----------------------------------------------------------------------------
function switchNavbarModule(modId) {
  activeModule = modId;

  // 1. Hentikan kamera dan loop auto-inspeksi saat berganti tab
  stopCamera();
  if (typeof isRingAuto !== 'undefined' && isRingAuto) toggleRingAuto();
  if (typeof isCastAuto !== 'undefined' && isCastAuto) toggleCastAuto();

  // 2. Perbarui status tombol tab navbar
  ['Ring', 'Bearing', 'Plastic'].forEach(m => {
    const btn = document.getElementById(`btnNav${m}`);
    if (btn) btn.classList.toggle('active', m.toLowerCase() === modId);
  });

  // 3. Tampilkan kontainer workspace yang sesuai
  const wsRing = document.getElementById('workspaceRing');
  const wsBearing = document.getElementById('workspaceBearing');
  const wsPlastic = document.getElementById('workspacePlastic');

  if (wsRing) wsRing.classList.toggle('active', modId === 'ring');
  if (wsBearing) wsBearing.classList.toggle('active', modId === 'bearing');
  if (wsPlastic) wsPlastic.classList.toggle('active', modId === 'plastic');
}

// -----------------------------------------------------------------------------
// Universal API Inspection Dispatcher
// -----------------------------------------------------------------------------
function submitModuleInference(modId, payload, isJson) {
  const options = { method: 'POST' };
  if (isJson) {
    options.headers = { 'Content-Type': 'application/json' };
    options.body = JSON.stringify(payload);
  } else {
    options.body = payload;
  }

  fetch(`/api/v1/inspect/${modId}`, options)
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        if (modId === 'ring' && typeof renderRingResults === 'function') {
          renderRingResults(data);
        } else if (modId === 'bearing' && typeof renderCastResults === 'function') {
          renderCastResults(data);
        } else if (modId === 'plastic' && typeof renderPlasticResults === 'function') {
          renderPlasticResults(data);
        }
      } else {
        alert(`Peringatan Inspeksi: ${data.error_message || data.message || 'Gagal memproses inspeksi citra'}`);
      }
    })
    .catch(err => {
      alert('Kesalahan Koneksi Inspeksi: ' + err);
    });
}

// -----------------------------------------------------------------------------
// Utilitas Manajemen Hardware Kamera
// -----------------------------------------------------------------------------
function startCamera(videoElementId) {
  stopCamera();
  navigator.mediaDevices
    .getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    })
    .then(stream => {
      activeVideoStream = stream;
      const vid = document.getElementById(videoElementId);
      if (vid) vid.srcObject = stream;
      populateCameraList();
    })
    .catch(err => alert("Aliran kamera tidak dapat diakses: " + err));
}

function stopCamera() {
  if (activeVideoStream) {
    activeVideoStream.getTracks().forEach(track => track.stop());
    activeVideoStream = null;
  }
}

function switchCameraFeed(mod) {
  startCamera(mod === 'ring' ? 'ringVideo' : 'castVideo');
}

function populateCameraList() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  navigator.mediaDevices.enumerateDevices().then(devices => {
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    const select = document.getElementById('cameraSelect');
    if (!select) return;
    select.innerHTML = '';
    videoDevices.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.text = dev.label || `Kamera ${idx + 1}`;
      select.appendChild(opt);
    });
  });
}

function onCameraSelectChange(deviceId) {
  stopCamera();
  navigator.mediaDevices
    .getUserMedia({
      video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    })
    .then(stream => {
      activeVideoStream = stream;
      const targetId = activeModule === 'ring' ? 'ringVideo' : 'castVideo';
      const vid = document.getElementById(targetId);
      if (vid) vid.srcObject = stream;
    })
    .catch(err => alert("Gagal beralih ke kamera yang dipilih: " + err));
}

function captureCanvasFrame(videoElementId) {
  const vid = document.getElementById(videoElementId);
  if (!vid || !vid.videoWidth) return null;
  const canvas = document.createElement('canvas');
  canvas.width = vid.videoWidth;
  canvas.height = vid.videoHeight;
  const ctx = canvas.getContext('2d');
  // Mirroring horizontal agar hasil tangkapan sesuai tampilan kamera cermin
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.92);
}

// -----------------------------------------------------------------------------
// Utilitas Helper DOM Tampilan
// -----------------------------------------------------------------------------
function setImg(imgId, phId, base64Url) {
  const img = document.getElementById(imgId);
  const ph = document.getElementById(phId);
  if (!img) return;

  if (base64Url) {
    img.src = base64Url;
    img.style.display = 'block';
    if (ph) ph.style.display = 'none';
  } else {
    img.style.display = 'none';
    if (ph) ph.style.display = 'block';
  }
}

// -----------------------------------------------------------------------------
// Lightbox Zoom Modal
// -----------------------------------------------------------------------------
function zoomImg(imgElementId, title) {
  const el = document.getElementById(imgElementId);
  if (!el || el.style.display === 'none' || !el.src) return;
  const modal = document.getElementById('zoomModal');
  const modalImg = document.getElementById('zoomImg');
  const modalTitle = document.getElementById('zoomTitle');
  if (modal && modalImg) {
    modalImg.src = el.src;
    if (modalTitle) modalTitle.innerText = title || 'Pratinjau Citra';
    modal.classList.add('open');
  }
}

function closeZoomModal() {
  const modal = document.getElementById('zoomModal');
  if (modal) modal.classList.remove('open');
}

function closeZoom(e) {
  if (e && e.target && e.target.id === 'zoomModal') {
    closeZoomModal();
  }
}

function closeZoomDirect() {
  closeZoomModal();
}

// Close modals on Escape key
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    closeZoomModal();
  }
});
