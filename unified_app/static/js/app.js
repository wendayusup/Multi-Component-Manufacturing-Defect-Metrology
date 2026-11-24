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
let selectedCameraDeviceId = null;

function startCamera(videoElementId, preferredDeviceId) {
  stopCamera();
  const targetDeviceId = preferredDeviceId || selectedCameraDeviceId;

  const constraints = {
    video: targetDeviceId 
      ? { deviceId: { exact: targetDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  };

  const bindStream = (stream) => {
    activeVideoStream = stream;
    const vid = document.getElementById(videoElementId);
    if (vid) {
      vid.srcObject = stream;
      vid.play().catch(() => {});
    }
    const track = stream.getVideoTracks()[0];
    if (track) {
      const s = track.getSettings();
      if (s.deviceId) selectedCameraDeviceId = s.deviceId;
    }
    populateCameraList();
  };

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Akses kamera membutuhkan HTTPS atau koneksi localhost.");
    return;
  }

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then(bindStream)
    .catch(err => {
      // Fallback jika constraint resolusi tinggi atau deviceId exact ditolak perangkat
      navigator.mediaDevices
        .getUserMedia({ video: true, audio: false })
        .then(bindStream)
        .catch(err2 => alert("Aliran kamera tidak dapat diakses: " + err2.message));
    });
}

function stopCamera() {
  if (activeVideoStream) {
    activeVideoStream.getTracks().forEach(track => track.stop());
    activeVideoStream = null;
  }
}

function populateCameraList() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
  navigator.mediaDevices.enumerateDevices().then(devices => {
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    const selects = [
      document.getElementById('ringCamSelect'),
      document.getElementById('cameraSelect')
    ].filter(Boolean);

    if (selects.length === 0) return;

    selects.forEach(sel => {
      const prevVal = sel.value || selectedCameraDeviceId;
      sel.innerHTML = '';
      if (videoDevices.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.text = 'Tidak ada kamera terdeteksi';
        sel.appendChild(opt);
        return;
      }
      videoDevices.forEach((dev, idx) => {
        const opt = document.createElement('option');
        opt.value = dev.deviceId;
        opt.text = dev.label || `Kamera ${idx + 1} (${dev.deviceId ? dev.deviceId.substring(0, 8) + '...' : 'USB/Webcam'})`;
        if (dev.deviceId && (dev.deviceId === prevVal || dev.deviceId === selectedCameraDeviceId)) {
          opt.selected = true;
        }
        sel.appendChild(opt);
      });
      if (!selectedCameraDeviceId && sel.value) {
        selectedCameraDeviceId = sel.value;
      }
    });
  }).catch(() => {});
}

// Deteksi otomatis saat kamera baru dicolokkan / dicabut
if (typeof navigator !== 'undefined' && navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener('devicechange', () => {
    populateCameraList();
  });
}

function onCameraDeviceChanged(mod, deviceId) {
  if (!deviceId) return;
  selectedCameraDeviceId = deviceId;
  const vidId = mod === 'ring' ? 'ringVideo' : 'castVideo';
  startCamera(vidId, deviceId);
}

function onCameraSelectChange(deviceId) {
  onCameraDeviceChanged('ring', deviceId);
}

function switchCameraFeed(mod) {
  const sel = document.getElementById(mod === 'ring' ? 'ringCamSelect' : 'cameraSelect');
  if (sel && sel.options.length > 1) {
    const nextIdx = (sel.selectedIndex + 1) % sel.options.length;
    sel.selectedIndex = nextIdx;
    onCameraDeviceChanged(mod, sel.value);
  } else {
    startCamera(mod === 'ring' ? 'ringVideo' : 'castVideo');
  }
}

function captureCanvasFrame(videoElementId) {
  const vid = document.getElementById(videoElementId);
  if (!vid || !vid.videoWidth) return null;

  const vw = vid.videoWidth;
  const vh = vid.videoHeight;

  // Pemotongan bujur sangkar di bagian tengah citra video
  const cropSize = Math.min(vw, vh);
  const sx = Math.max(0, Math.floor((vw - cropSize) / 2));
  const sy = Math.max(0, Math.floor((vh - cropSize) / 2));

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  // Efek cermin horizontal (mirroring) agar natural bagi operator
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(vid, sx, sy, cropSize, cropSize, 0, 0, 512, 512);

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
