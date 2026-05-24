// Alphafolio Quant Console — vanilla JS, fetch-based.
// Auth: server-side reads API_SECRET_KEY from secrets/alphafolio.env.
// No browser-side key handling.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ---------------- Tabs ----------------
$$('.tab-btn').forEach(btn => {
  btn.onclick = () => {
    $$('.tab-btn').forEach(b => b.classList.remove('active'));
    $$('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $(`#tab-${btn.dataset.tab}`).classList.add('active');
    if (btn.dataset.tab === 'results') refreshResults();
  };
});

// ---------------- Logging ----------------
const logsEl = $('#logs');
function log(type, msg) {
  const ts = new Date().toLocaleTimeString();
  const line = document.createElement('div');
  line.className = `log-line ${type}`;
  line.textContent = `[${ts}] ${msg}`;
  logsEl.prepend(line);
}
$('#clearLogsBtn').onclick = () => { logsEl.innerHTML = ''; };

// ---------------- Backfill buttons ----------------
$$('button.bf').forEach(btn => {
  btn.onclick = async () => {
    const endpoint = btn.dataset.endpoint;
    const params = btn.dataset.params || '';
    const url = `/backfill/${endpoint}${params ? '?' + params : ''}`;
    btn.disabled = true;
    btn.classList.add('running');
    const origText = btn.textContent;
    btn.textContent = '⏳ ' + origText;
    log('run', `POST ${endpoint} ${params ? '?' + params : ''}`);
    const t0 = Date.now();
    try {
      const r = await fetch(url, { method: 'POST' });
      const data = await r.json();
      const sec = ((Date.now() - t0) / 1000).toFixed(1);
      if (r.ok) {
        log('ok', `✓ ${endpoint} (${sec}s): ${JSON.stringify(data).slice(0, 200)}`);
      } else {
        log('err', `✗ ${endpoint} (${sec}s): ${r.status} ${JSON.stringify(data).slice(0, 300)}`);
      }
    } catch (e) {
      log('err', `✗ ${endpoint}: ${e.message}`);
    } finally {
      btn.disabled = false;
      btn.classList.remove('running');
      btn.textContent = origText;
    }
  };
});

// ---------------- Grade generation form ----------------
$('#genGradesForm').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {
    country: fd.get('country'),
    start_date: fd.get('start_date'),
    end_date: fd.get('end_date'),
    skip_existing: fd.get('skip_existing') === 'on',
  };
  const btn = e.target.querySelector('button');
  btn.disabled = true; btn.textContent = '⏳ Generating...';
  log('run', `POST /backtest/generate-grades ${JSON.stringify(body)}`);
  const t0 = Date.now();
  try {
    const r = await fetch('/backtest/generate-grades', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    const sec = ((Date.now() - t0) / 1000).toFixed(1);
    if (r.ok) {
      log('ok', `✓ generate-grades (${sec}s): processed=${data.processed_count} skipped=${data.skipped_count} failed=${data.failed_count}`);
    } else {
      log('err', `✗ generate-grades (${sec}s): ${JSON.stringify(data).slice(0, 300)}`);
    }
  } catch (e) {
    log('err', `✗ generate-grades: ${e.message}`);
  } finally {
    btn.disabled = false; btn.textContent = 'Generate Grades';
  }
};

// ---------------- Backtest form ----------------
$('#runBacktestForm').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const body = {
    country: fd.get('country'),
    start_date: fd.get('start_date'),
    end_date: fd.get('end_date'),
    initial_cash: parseFloat(fd.get('initial_cash')),
    top_n: parseInt(fd.get('top_n')),
    rebal_freq_days: parseInt(fd.get('rebal_freq_days')),
    grades_filter: fd.get('grades_filter').split(',').map(s => s.trim()).filter(Boolean),
    commission_rate: parseFloat(fd.get('commission_rate')),
    slippage_rate: parseFloat(fd.get('slippage_rate')),
  };
  const btn = e.target.querySelector('button');
  btn.disabled = true; btn.textContent = '⏳ Running...';
  log('run', `POST /backtest/run ${body.country} ${body.start_date}~${body.end_date}`);
  const t0 = Date.now();
  try {
    const r = await fetch('/backtest/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    const sec = ((Date.now() - t0) / 1000).toFixed(1);
    if (r.ok) {
      log('ok', `✓ backtest ${data.run_id} (${sec}s): NAV=${data.metrics.final_nav.toLocaleString()} return=${(data.metrics.total_return*100).toFixed(2)}% sharpe=${data.metrics.sharpe}`);
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      $$('.tab-pane').forEach(p => p.classList.remove('active'));
      $('.tab-btn[data-tab="results"]').classList.add('active');
      $('#tab-results').classList.add('active');
      refreshResults();
    } else {
      log('err', `✗ backtest (${sec}s): ${JSON.stringify(data).slice(0, 300)}`);
    }
  } catch (e) {
    log('err', `✗ backtest: ${e.message}`);
  } finally {
    btn.disabled = false; btn.textContent = 'Run Backtest';
  }
};

// ---------------- Results table ----------------
$('#refreshResultsBtn').onclick = refreshResults;

async function refreshResults() {
  log('info', 'Loading results...');
  try {
    const r = await fetch('/backtest/results?limit=100');
    const data = await r.json();
    const tbody = $('#resultsTable tbody');
    tbody.innerHTML = '';
    data.runs.forEach(run => {
      const m = run.metrics ? (typeof run.metrics === 'string' ? JSON.parse(run.metrics) : run.metrics) : {};
      const tr = document.createElement('tr');
      const ret = m.total_return != null ? (m.total_return * 100).toFixed(2) + '%' : '-';
      const cagr = m.cagr != null ? (m.cagr * 100).toFixed(2) + '%' : '-';
      const sharpe = m.sharpe != null ? m.sharpe.toFixed(2) : '-';
      const mdd = m.mdd != null ? (m.mdd * 100).toFixed(2) + '%' : '-';
      const nav = m.final_nav != null ? Math.round(m.final_nav).toLocaleString() : '-';
      const started = run.started_at ? new Date(run.started_at).toLocaleString() : '-';
      tr.innerHTML = `
        <td><code>${run.run_id}</code></td>
        <td>${run.country}</td>
        <td>${run.start_date} ~ ${run.end_date}</td>
        <td>${statusBadge(run.status)}</td>
        <td>${nav}</td>
        <td class="${classFor(m.total_return)}">${ret}</td>
        <td class="${classFor(m.cagr)}">${cagr}</td>
        <td>${sharpe}</td>
        <td class="neg">${mdd}</td>
        <td>${started}</td>
        <td>
          <button class="link" onclick="showDetail('${run.run_id}')">View</button>
          <button class="link" onclick="deleteRun('${run.run_id}')" style="color:#f85149">Del</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
    log('info', `Loaded ${data.runs.length} runs`);
  } catch (e) {
    log('err', `✗ list results: ${e.message}`);
  }
}

function statusBadge(s) {
  if (s === 'completed') return '<span style="color:#56d364">✓ done</span>';
  if (s === 'running') return '<span style="color:#d29922">⏳ running</span>';
  if (s === 'failed') return '<span style="color:#f85149">✗ failed</span>';
  return s;
}
function classFor(v) {
  if (v == null) return '';
  return v >= 0 ? 'pos' : 'neg';
}

window.showDetail = async (runId) => {
  log('info', `Loading detail for ${runId}`);
  try {
    const r = await fetch(`/backtest/results/${runId}?include_nav=true&include_trades=true`);
    const data = await r.json();
    $('#detailPanel').classList.remove('hidden');
    $('#detailRunId').textContent = runId;
    const m = typeof data.metrics === 'string' ? JSON.parse(data.metrics) : (data.metrics || {});
    $('#detailMetrics').innerHTML = `
      <div class="metric-grid">
        ${metricCard('Total Return', m.total_return, true)}
        ${metricCard('CAGR', m.cagr, true)}
        ${metricCard('Sharpe', m.sharpe)}
        ${metricCard('Sortino', m.sortino)}
        ${metricCard('MDD', m.mdd, true)}
        ${metricCard('Calmar', m.calmar)}
        ${metricCard('Win Rate', m.win_rate, true)}
        ${metricCard('Final NAV', m.final_nav)}
      </div>
    `;
    const navTbody = $('#navTable tbody');
    navTbody.innerHTML = '';
    (data.nav_history || []).slice(-30).forEach(n => {
      navTbody.innerHTML += `<tr><td>${n.date}</td><td>${Math.round(n.nav).toLocaleString()}</td><td>${n.holdings_count}</td><td>${Math.round(n.cash).toLocaleString()}</td></tr>`;
    });
    const trTbody = $('#tradesTable tbody');
    trTbody.innerHTML = '';
    (data.trades || []).slice(-30).forEach(t => {
      const cls = t.action === 'BUY' ? 'pos' : 'neg';
      trTbody.innerHTML += `<tr><td>${t.date}</td><td>${t.symbol}</td><td class="${cls}">${t.action}</td><td>${t.shares}</td><td>${parseFloat(t.price).toFixed(2)}</td><td>${Math.round(t.amount).toLocaleString()}</td></tr>`;
    });
    log('ok', `Loaded detail for ${runId}: ${data.nav_history?.length || 0} nav, ${data.trades?.length || 0} trades`);
  } catch (e) {
    log('err', `✗ detail: ${e.message}`);
  }
};

function metricCard(label, v, isPercent) {
  if (v == null) return `<div class="metric-card"><div class="label">${label}</div><div class="value">-</div></div>`;
  const display = isPercent && Math.abs(v) <= 10 ? (v * 100).toFixed(2) + '%' : (typeof v === 'number' ? v.toLocaleString() : v);
  const cls = typeof v === 'number' ? (v >= 0 ? 'pos' : 'neg') : '';
  return `<div class="metric-card"><div class="label">${label}</div><div class="value ${cls}">${display}</div></div>`;
}

window.deleteRun = async (runId) => {
  if (!confirm(`Delete ${runId}?`)) return;
  try {
    const r = await fetch(`/backtest/results/${runId}`, { method: 'DELETE' });
    if (r.ok) { log('ok', `Deleted ${runId}`); refreshResults(); }
    else { log('err', `Failed to delete ${runId}: ${r.status}`); }
  } catch (e) { log('err', `✗ delete: ${e.message}`); }
};

// Initial state
log('info', 'Console ready. Auth uses API_SECRET_KEY from secrets/alphafolio.env.');
