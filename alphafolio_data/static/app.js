// Alphafolio Data Console — vanilla JS.
// Auth: server reads API_SECRET_KEY from env. Browser sends no header.

const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ---------------- Tabs ----------------
$$('.tab-btn').forEach(btn => {
  btn.onclick = () => {
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'runs') refreshRuns();
  };
});

// ---------------- Manual collection buttons (existing pattern) ----------------
$$('button.bf').forEach(btn => {
  btn.onclick = async () => {
    btn.disabled = true; btn.classList.add('running');
    const txt = btn.textContent;
    btn.textContent = '⏳ ' + txt;
    try {
      const r = await fetch(`/${btn.dataset.endpoint}`, {method: 'POST'});
      const data = await r.json();
      console.log(`${btn.dataset.endpoint}:`, data);
      alert(r.ok ? '✓ 완료' : `✗ ${r.status}: ${JSON.stringify(data).slice(0,200)}`);
    } catch (e) {
      alert(`✗ ${e.message}`);
    } finally {
      btn.disabled = false; btn.classList.remove('running');
      btn.textContent = txt;
    }
  };
});

// ---------------- Pipeline form ----------------
$('#pipelineForm').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {
    country: fd.get('country'),
    start_date: fd.get('start_date') || null,
    end_date:   fd.get('end_date')   || null,
    option_top_n:    parseInt(fd.get('option_top_n')),
    backtest_top_n:  parseInt(fd.get('backtest_top_n')),
    rebal_freq_days: parseInt(fd.get('rebal_freq_days')),
    initial_cash:    parseFloat(fd.get('initial_cash')),
  };
  const btn = e.target.querySelector('button[type=submit]');
  btn.disabled = true; btn.textContent = '⏳ 시작 중...';
  try {
    const r = await fetch('/orchestrator/runs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) {
      alert(`✗ ${r.status}: ${JSON.stringify(data).slice(0,300)}`);
    } else {
      $('#activeRun').innerHTML = `
        <strong>${data.run_id}</strong> 시작됨
        (${data.country} ${data.start_date} ~ ${data.end_date})
      `;
      openDagPanel(data.run_id);
    }
  } catch (e) {
    alert(`✗ ${e.message}`);
  } finally {
    btn.disabled = false; btn.textContent = '파이프라인 시작';
  }
};

// ---------------- Runs list ----------------
$('#refreshRunsBtn').onclick = refreshRuns;

async function refreshRuns() {
  const r = await fetch('/orchestrator/runs?limit=100');
  const data = await r.json();
  const tbody = $('#runsTable tbody');
  tbody.innerHTML = '';
  data.runs.forEach(run => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><code>${run.run_id}</code></td>
      <td>${run.country}</td>
      <td>${run.start_date} ~ ${run.end_date}</td>
      <td>${statusBadge(run.status)}</td>
      <td>${run.started_at ? new Date(run.started_at).toLocaleString() : '-'}</td>
      <td><button onclick="openDagPanel('${run.run_id}')">View</button></td>
    `;
    tbody.appendChild(tr);
  });
}

function statusBadge(s) {
  const colors = {
    completed: '#56d364', running: '#d29922',
    failed: '#f85149',  cancelled: '#6e7681',
    pending: '#8b949e',
  };
  return `<span style="color:${colors[s]||'#c9d1d9'}">${s}</span>`;
}

// ---------------- DAG side panel ----------------
let _activeRunId = null;
let _logCursor = 0;
let _pollHandle = null;

window.openDagPanel = (runId) => {
  _activeRunId = runId;
  _logCursor = 0;
  $('#dagPanel').classList.remove('hidden');
  $('#dagRunId code').textContent = runId;
  $('#dagTasks').innerHTML = '';
  $('#dagLogs').innerHTML = '';
  pollDag();
  if (_pollHandle) clearInterval(_pollHandle);
  _pollHandle = setInterval(pollDag, 2000);
};

$('#dagCloseBtn').onclick = () => {
  $('#dagPanel').classList.add('hidden');
  if (_pollHandle) { clearInterval(_pollHandle); _pollHandle = null; }
  _activeRunId = null;
};

$('#dagRefreshBtn').onclick = () => pollDag();

$('#dagResumeBtn').onclick = async () => {
  if (!_activeRunId) return;
  if (!confirm(`${_activeRunId} 재시작?`)) return;
  await fetch(`/orchestrator/runs/${_activeRunId}/resume`, {method:'POST'});
  pollDag();
};

$('#dagCancelBtn').onclick = async () => {
  if (!_activeRunId) return;
  if (!confirm(`${_activeRunId} 중단?`)) return;
  await fetch(`/orchestrator/runs/${_activeRunId}/cancel`, {method:'POST'});
  pollDag();
};

async function pollDag() {
  if (!_activeRunId) return;
  try {
    const [runR, logsR] = await Promise.all([
      fetch(`/orchestrator/runs/${_activeRunId}`),
      fetch(`/orchestrator/runs/${_activeRunId}/logs?after_id=${_logCursor}&limit=300`),
    ]);
    if (runR.ok) {
      const d = await runR.json();
      renderTasks(d.tasks);
    }
    if (logsR.ok) {
      const l = await logsR.json();
      appendLogs(l.logs);
      _logCursor = l.next_after_id;
    }
  } catch (e) {
    console.error(e);
  }
}

function renderTasks(tasks) {
  const ol = $('#dagTasks');
  ol.innerHTML = '';
  tasks.forEach(t => {
    const li = document.createElement('li');
    li.className = `status-${t.status}`;
    const dur = t.duration_seconds ? ` (${t.duration_seconds.toFixed(1)}s)` : '';
    const err = t.error ? ` — ${t.error.slice(0,80)}` : '';
    li.innerHTML = `
      <span class="name">${t.name}</span>
      <span class="meta">${t.status}${dur}${err}</span>
    `;
    ol.appendChild(li);
  });
}

function appendLogs(logs) {
  if (!logs || !logs.length) return;
  const pre = $('#dagLogs');
  logs.forEach(l => {
    const ts = new Date(l.ts).toLocaleTimeString();
    const tid = l.task_id ? `[${l.task_id}] ` : '';
    const div = document.createElement('div');
    div.className = `log-line ${l.level}`;
    div.textContent = `${ts} ${tid}${l.message}`;
    pre.appendChild(div);
  });
  pre.scrollTop = pre.scrollHeight;
}

// Initial: auto-open most recent active run's DAG panel so logs are visible
async function autoOpenActiveRun() {
  try {
    const r = await fetch('/orchestrator/runs?limit=5');
    const data = await r.json();
    const active = data.runs.find(x => x.status === 'running' || x.status === 'pending');
    // Skip if same run already open (avoid resetting poll cursor)
    if (active && active.run_id !== _activeRunId) {
      console.log('Auto-opening active run:', active.run_id);
      openDagPanel(active.run_id);
    }
  } catch (e) {
    console.error('autoOpenActiveRun failed:', e);
  }
}

refreshRuns();
autoOpenActiveRun();
// Re-check every 10s for a newly-started run (no-op if same id already open)
setInterval(autoOpenActiveRun, 10000);
