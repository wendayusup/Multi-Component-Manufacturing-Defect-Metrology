/**
 * HEALTH_MONITOR.JS — Shared Live Server & Network Signal Indicator
 * Polls /api/v1/health every 5 seconds on Home, Dashboard, and Inspection.
 * Shows 4-bar signal strength + server ping (ms) + instant disconnect detection.
 */
(function () {
  const POLL_INTERVAL_MS = 5000;
  const TIMEOUT_MS = 4000;

  function renderSignalSvg(barsLit, color) {
    const heights = [4, 7, 10, 13];
    const rects = heights.map((h, idx) => {
      const x = idx * 4;
      const y = 14 - h;
      const fill = idx < barsLit ? color : '#CBD5E1';
      return `<rect x="${x}" y="${y}" width="2.6" height="${h}" rx="0.8" fill="${fill}"></rect>`;
    }).join('');
    return `<svg width="16" height="14" viewBox="0 0 16 14" aria-hidden="true" style="flex-shrink:0;">${rects}</svg>`;
  }

  function updateBadgeUI(badge, state) {
    if (!badge) return;
    badge.style.background = state.bg;
    badge.style.color = state.color;
    badge.title = state.tooltip;
    badge.innerHTML = `${renderSignalSvg(state.bars, state.color)}<span>${state.label}</span>`;
  }

  async function checkServerHealth() {
    const badge =
      document.getElementById('homeHealthBadge') ||
      document.getElementById('dashHealthBadge') ||
      document.getElementById('inspectHealthBadge');

    if (!badge) return;

    if (!navigator.onLine) {
      updateBadgeUI(badge, {
        bars: 0,
        color: '#DC2626',
        bg: '#FEF2F2',
        label: 'Network Offline',
        tooltip: 'Koneksi jaringan perangkat terputus.'
      });
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    const t0 = performance.now();

    try {
      const res = await fetch('/api/v1/health', {
        cache: 'no-store',
        signal: controller.signal
      });
      clearTimeout(timer);
      const pingMs = Math.max(1, Math.round(performance.now() - t0));

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      let bars = 4;
      if (pingMs >= 400) bars = 2;
      else if (pingMs >= 150) bars = 3;

      const allReady = !!data.all_models_ready;
      const connInfo = navigator.connection && navigator.connection.effectiveType
        ? ` • ${navigator.connection.effectiveType.toUpperCase()}`
        : '';

      if (!allReady) {
        updateBadgeUI(badge, {
          bars: Math.min(bars, 2),
          color: '#D97706',
          bg: '#FFFBEB',
          label: `Partial Ready • ${pingMs} ms`,
          tooltip: `Server terhubung (${pingMs} ms${connInfo}), sebagian model belum siap.`
        });
      } else if (bars <= 2) {
        updateBadgeUI(badge, {
          bars,
          color: '#D97706',
          bg: '#FFFBEB',
          label: `Slow Server • ${pingMs} ms`,
          tooltip: `Server lambat (${pingMs} ms${connInfo}).`
        });
      } else {
        updateBadgeUI(badge, {
          bars,
          color: '#059669',
          bg: '#ECFDF5',
          label: `System Healthy • ${pingMs} ms`,
          tooltip: `Server aktif (${pingMs} ms${connInfo}) • Semua model siap.`
        });
      }
    } catch (err) {
      clearTimeout(timer);
      updateBadgeUI(badge, {
        bars: 0,
        color: '#DC2626',
        bg: '#FEF2F2',
        label: 'Server Offline',
        tooltip: 'Server tidak merespons. Pastikan terminal menjalankan: python run.py'
      });
    }
  }

  window.checkGlobalServerHealth = checkServerHealth;

  document.addEventListener('DOMContentLoaded', () => {
    checkServerHealth();
    setInterval(checkServerHealth, POLL_INTERVAL_MS);
  });

  window.addEventListener('online', checkServerHealth);
  window.addEventListener('offline', checkServerHealth);
})();
