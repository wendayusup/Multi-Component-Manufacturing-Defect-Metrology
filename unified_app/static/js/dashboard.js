/**
 * DASHBOARD.JS — Live Health Polling for Model Status Badges
 */
async function loadDashboardHealth() {
  const t0 = performance.now();
  const badge = document.getElementById('dashHealthBadge');
  const pingEl = document.getElementById('kpiPing');

  try {
    const [healthRes, modRes] = await Promise.all([
      fetch('/api/v1/health'),
      fetch('/api/v1/modules')
    ]);
    const health = await healthRes.json();
    const modData = await modRes.json();
    const pingMs = Math.round(performance.now() - t0);

    const modules = modData.modules || {};
    const keys = Object.keys(modules);
    const readyCount = keys.filter(k => modules[k].status === 'ready').length;

    document.getElementById('kpiReady').textContent = `${readyCount} / ${keys.length}`;
    document.getElementById('kpiReadySub').textContent = health.all_models_ready
      ? 'All CNN adapters active & ready'
      : 'Some modules unavailable';

    pingEl.textContent = `${pingMs} ms`;
    if (window.checkGlobalServerHealth) window.checkGlobalServerHealth();

    const badgeMap = { ring: 'badgeRing', bearing: 'badgeBearing', plastic: 'badgePlastic' };
    Object.entries(badgeMap).forEach(([modId, elId]) => {
      const el = document.getElementById(elId);
      const m = modules[modId];
      if (!el || !m) return;
      const isReady = m.status === 'ready';
      el.textContent = (m.status || 'unknown').toUpperCase();
      el.className = `badge-ready ${isReady ? '' : 'badge-err'}`;
    });
  } catch (err) {
    badge.textContent = 'API Offline';
    badge.style.background = '#FEF2F2';
    badge.style.color = '#DC2626';
    pingEl.textContent = 'ERR';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadDashboardHealth();
});
