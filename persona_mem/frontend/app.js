// ── Shared utilities for PersonaMem frontend ──────────────────────────────────

const TOKEN_KEY = 'token';

/**
 * Authenticated API fetch utility
 */
async function apiFetch(path, method = 'GET', body = null) {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) {
    window.location.href = '/';
    throw new Error('Not authenticated');
  }

  const opts = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
  };

  if (body !== null) {
    opts.body = JSON.stringify(body);
  }

  const res = await fetch(path, opts);

  if (res.status === 401) {
    localStorage.clear();
    window.location.href = '/';
    throw new Error('Session expired');
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }

  // Handle empty responses
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch (_) {
    return text;
  }
}

/**
 * Show a toast notification
 * @param {string} msg
 * @param {'success'|'error'|''} type
 */
function showToast(msg, type = '') {
  const toast = document.getElementById('toast');
  if (!toast) return;
  toast.textContent = msg;
  toast.className = 'toast ' + type;
  toast.style.display = 'block';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.style.display = 'none'; }, 3200);
}

/**
 * Escape HTML to prevent XSS
 */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Human-readable relative time
 */
function timeAgo(isoString) {
  if (!isoString) return '—';
  const date = new Date(isoString);
  const now = new Date();
  const diff = Math.floor((now - date) / 1000);

  if (diff < 60)     return 'just now';
  if (diff < 3600)   return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400)  return Math.floor(diff / 3600) + 'h ago';
  if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';

  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/**
 * Format a number with K/M abbreviation
 */
function fmtNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}

/**
 * Debounce utility
 */
function debounce(fn, delay) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

/**
 * Copy text to clipboard with fallback
 */
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    showToast('Copied!', 'success');
  } catch (_) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    showToast('Copied!', 'success');
  }
}
