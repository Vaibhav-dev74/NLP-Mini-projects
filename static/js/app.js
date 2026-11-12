/**
 * Dialogue Classifier 2.0 - Client Application Engine
 * Handles Theme Toggling, Live Classification, Chart.js Visualizations,
 * Batch Processing, History Filtering & Real-time Analytics.
 */

// =====================================================================
// 1. THEME SWITCHER
// =====================================================================
function initTheme() {
  const savedTheme = localStorage.getItem('dc_theme') || 'light';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);

  const toggleBtn = document.getElementById('themeToggleBtn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const target = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', target);
      localStorage.setItem('dc_theme', target);
      updateThemeIcon(target);

      // Trigger chart re-render if on analytics or classifier page
      if (window.probabilityChart) window.probabilityChart.update();
      if (window.categoryChart) window.categoryChart.update();
      if (window.sentimentChart) window.sentimentChart.update();
    });
  }
}

function updateThemeIcon(theme) {
  const toggleBtn = document.getElementById('themeToggleBtn');
  if (!toggleBtn) return;
  if (theme === 'dark') {
    toggleBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      </svg>
    `;
  } else {
    toggleBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      </svg>
    `;
  }
}

// Mobile Sidebar Toggle
function initMobileSidebar() {
  const toggleBtn = document.getElementById('mobileSidebarToggle');
  const sidebar = document.querySelector('.sidebar');
  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });
  }
}

// =====================================================================
// 2. TEXT CLASSIFIER CONTROLLER
// =====================================================================
let probabilityChart = null;

function initClassifierWorkspace() {
  const form = document.getElementById('classifierForm');
  const textInput = document.getElementById('dialogueInput');
  const modeSelect = document.getElementById('modeSelect');
  const resultsCard = document.getElementById('resultsContainer');
  const loadingIndicator = document.getElementById('classifyLoading');
  const errorContainer = document.getElementById('classifyError');

  // Sample sentence chips
  document.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      textInput.value = chip.getAttribute('data-text');
      textInput.focus();
    });
  });

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = textInput.value.trim();
    if (!text) {
      showError('Please enter some text or dialogue to classify.');
      return;
    }

    const mode = modeSelect.value;
    clearError();
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (resultsCard) resultsCard.style.display = 'none';

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, mode, save_history: true })
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to classify text.');
      }

      renderClassificationResults(data);
    } catch (err) {
      showError(err.message);
    } finally {
      if (loadingIndicator) loadingIndicator.style.display = 'none';
    }
  });

  function showError(msg) {
    if (errorContainer) {
      errorContainer.textContent = msg;
      errorContainer.style.display = 'block';
    }
  }

  function clearError() {
    if (errorContainer) {
      errorContainer.textContent = '';
      errorContainer.style.display = 'none';
    }
  }
}

function renderClassificationResults(data) {
  const container = document.getElementById('resultsContainer');
  if (!container) return;

  // 1. Hero Category & Badge
  const categoryElem = document.getElementById('resultCategory');
  const categoryBadge = document.getElementById('resultCategoryBadge');
  const confValue = document.getElementById('resultConfidenceValue');
  const confBar = document.getElementById('resultConfidenceBar');
  const modeBadge = document.getElementById('resultModeBadge');

  if (categoryElem) categoryElem.textContent = data.category;
  if (categoryBadge) {
    categoryBadge.textContent = data.category;
    categoryBadge.className = `badge badge-${data.category.toLowerCase().replace('/', '-')}`;
  }

  if (modeBadge) {
    const modeLabels = {
      'hybrid': 'Hybrid (Rules + ML)',
      'rule_based_enhanced': 'Enhanced Rule Engine',
      'rule_based_original': 'Original v1.0 Legacy Engine',
      'machine_learning': 'Machine Learning (TF-IDF)'
    };
    modeBadge.textContent = modeLabels[data.mode] || data.mode;
  }

  // 1.1 User Intent Mapping
  const intentText = document.getElementById('resultIntentText');
  const intentDesc = document.getElementById('resultIntentDesc');
  const summaryBtn = document.getElementById('downloadSummaryBtn');

  if (data.user_intent) {
    if (intentText) {
      intentText.textContent = data.user_intent.specific_sub_intent || data.user_intent.primary_intent;
    }
    if (intentDesc) {
      intentDesc.textContent = data.user_intent.description || '';
    }
  }

  if (summaryBtn) {
    if (data.record_id) {
      summaryBtn.href = `/api/export/summary/${data.record_id}`;
      summaryBtn.style.display = 'inline-flex';
    } else {
      summaryBtn.style.display = 'none';
    }
  }

  // 2. Confidence / Probability
  if (data.is_rule_based) {
    if (confValue) confValue.textContent = 'Rule-Based Pattern Match';
    if (confBar) {
      confBar.style.width = '100%';
      confBar.style.background = 'linear-gradient(90deg, #3b82f6, #8b5cf6)';
    }
  } else if (data.confidence !== null && data.confidence !== undefined) {
    const pct = Math.round(data.confidence * 100);
    if (confValue) confValue.textContent = `${pct}% Model Probability`;
    if (confBar) {
      confBar.style.width = `${pct}%`;
      confBar.style.background = 'linear-gradient(90deg, #3b82f6, #10b981)';
    }
  } else {
    if (confValue) confValue.textContent = 'Confidence not applicable';
    if (confBar) confBar.style.width = '0%';
  }

  // 3. Explanation & Keywords
  const explElem = document.getElementById('resultExplanation');
  if (explElem) explElem.textContent = data.explanation || 'No detailed explanation generated.';

  const kwContainer = document.getElementById('resultKeywordsContainer');
  if (kwContainer) {
    kwContainer.innerHTML = '';
    const keywords = data.matched_keywords || [];
    if (keywords.length > 0) {
      keywords.forEach(kw => {
        const span = document.createElement('span');
        span.className = 'badge badge-statement';
        span.style.margin = '2px 4px';
        span.textContent = kw;
        kwContainer.appendChild(span);
      });
    } else {
      kwContainer.innerHTML = '<span style="color:var(--text-muted); font-size:0.85rem;">None detected</span>';
    }
  }

  // 4. Sentiment Result
  const sentBadge = document.getElementById('resultSentimentBadge');
  const sentPolarity = document.getElementById('resultSentimentPolarity');
  const sentExpl = document.getElementById('resultSentimentExplanation');
  if (data.sentiment) {
    const s = data.sentiment;
    if (sentBadge) {
      sentBadge.textContent = s.sentiment;
      sentBadge.className = `badge badge-sentiment-${s.sentiment.toLowerCase()}`;
    }
    if (sentPolarity) {
      sentPolarity.textContent = `Polarity Score: ${s.polarity_score > 0 ? '+' : ''}${s.polarity_score}`;
    }
    if (sentExpl) sentExpl.textContent = s.explanation;
  }

  // 5. Emotion Result
  const emoBadge = document.getElementById('resultEmotionBadge');
  const emoExpl = document.getElementById('resultEmotionExplanation');
  const emoDisclaimer = document.getElementById('resultEmotionDisclaimer');
  if (data.emotion) {
    const e = data.emotion;
    if (emoBadge) emoBadge.textContent = e.emotion;
    if (emoExpl) emoExpl.textContent = e.explanation;
    if (emoDisclaimer) emoDisclaimer.textContent = e.disclaimer;
  }

  // 6. Linguistic Stats
  if (data.statistics) {
    const st = data.statistics;
    const wordCount = document.getElementById('statWordCount');
    const charCount = document.getElementById('statCharCount');
    const sentCount = document.getElementById('statSentCount');
    const readTime = document.getElementById('statReadTime');
    const lexDiv = document.getElementById('statLexDiv');

    if (wordCount) wordCount.textContent = st.word_count;
    if (charCount) charCount.textContent = st.character_count;
    if (sentCount) sentCount.textContent = st.sentence_count;
    if (readTime) readTime.textContent = `${st.reading_time_seconds}s`;
    if (lexDiv) lexDiv.textContent = `${st.lexical_diversity}%`;
  }

  // 7. Probabilities Chart (Chart.js)
  renderProbabilityChart(data.probabilities, data.is_rule_based);

  // 8. Multi-sentence breakdown table
  renderSentenceBreakdown(data.sentence_breakdown);

  container.style.display = 'grid';
  container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderProbabilityChart(probabilities, isRuleBased) {
  const chartCanvas = document.getElementById('probabilityChart');
  const chartWrapper = document.getElementById('chartWrapper');
  const ruleNotice = document.getElementById('ruleOnlyNotice');

  if (!chartCanvas) return;

  if (isRuleBased || !probabilities) {
    if (chartWrapper) chartWrapper.style.display = 'none';
    if (ruleNotice) {
      ruleNotice.style.display = 'block';
      ruleNotice.textContent = 'Rule-based mode evaluated explicit syntactic and regex rules rather than generating machine learning probability distributions.';
    }
    return;
  }

  if (ruleNotice) ruleNotice.style.display = 'none';
  if (chartWrapper) chartWrapper.style.display = 'block';

  const labels = Object.keys(probabilities);
  const dataValues = Object.values(probabilities).map(v => Math.round(v * 100));

  if (probabilityChart) {
    probabilityChart.destroy();
  }

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const textColor = isDark ? '#9ca3af' : '#475569';
  const gridColor = isDark ? '#374151' : '#e2e8f0';

  const ctx = chartCanvas.getContext('2d');
  probabilityChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Probability (%)',
        data: dataValues,
        backgroundColor: [
          '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ec4899',
          '#8b5cf6', '#ef4444', '#14b8a6', '#6366f1', '#64748b'
        ],
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Probability: ${ctx.parsed.x}%`
          }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          ticks: { color: textColor, callback: (v) => `${v}%` },
          grid: { color: gridColor }
        },
        y: {
          ticks: { color: textColor, font: { weight: '600' } },
          grid: { display: false }
        }
      }
    }
  });
}

function renderSentenceBreakdown(breakdown) {
  const tableContainer = document.getElementById('sentenceBreakdownContainer');
  const tbody = document.getElementById('sentenceBreakdownBody');
  if (!tableContainer || !tbody) return;

  if (!breakdown || breakdown.length <= 1) {
    tableContainer.style.display = 'none';
    return;
  }

  tbody.innerHTML = '';
  breakdown.forEach((item, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td style="max-width:320px;"><em>"${escapeHtml(item.sentence)}"</em></td>
      <td><span class="badge badge-${item.category.toLowerCase().replace('/', '-')}">${item.category}</span></td>
      <td style="font-size:0.85rem; color:var(--text-secondary);">${escapeHtml(item.explanation)}</td>
    `;
    tbody.appendChild(tr);
  });
  tableContainer.style.display = 'block';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// =====================================================================
// 3. BATCH PROCESSOR CONTROLLER
// =====================================================================
function initBatchWorkspace() {
  const uploadForm = document.getElementById('batchUploadForm');
  const fileInput = document.getElementById('batchFileInput');
  const inspectStep = document.getElementById('inspectStep');
  const processBtn = document.getElementById('executeBatchBtn');
  const columnSelect = document.getElementById('textColumnSelect');
  const batchLoading = document.getElementById('batchLoading');
  const batchResults = document.getElementById('batchResults');

  let currentTempFile = null;

  if (fileInput) {
    fileInput.addEventListener('change', async () => {
      const file = fileInput.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      try {
        if (batchLoading) batchLoading.style.display = 'block';
        const res = await fetch('/api/batch/inspect', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Inspection failed.');

        currentTempFile = data.temp_filename;
        if (columnSelect) {
          columnSelect.innerHTML = '';
          data.columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            if (col === data.suggested_column) opt.selected = true;
            columnSelect.appendChild(opt);
          });
        }
        if (inspectStep) inspectStep.style.display = 'block';
      } catch (err) {
        alert('CSV Inspection Error: ' + err.message);
      } finally {
        if (batchLoading) batchLoading.style.display = 'none';
      }
    });
  }

  if (processBtn) {
    processBtn.addEventListener('click', async () => {
      if (!currentTempFile) {
        alert('Please select a valid CSV first.');
        return;
      }
      const textColumn = columnSelect ? columnSelect.value : '';
      const mode = document.getElementById('batchModeSelect')?.value || 'hybrid';

      const formData = new FormData();
      formData.append('temp_filename', currentTempFile);
      formData.append('text_column', textColumn);
      formData.append('mode', mode);

      try {
        if (batchLoading) batchLoading.style.display = 'block';
        if (batchResults) batchResults.style.display = 'none';

        const res = await fetch('/api/classify/batch', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Batch processing failed.');

        renderBatchResults(data);
      } catch (err) {
        alert('Batch Processing Error: ' + err.message);
      } finally {
        if (batchLoading) batchLoading.style.display = 'none';
      }
    });
  }
}

function renderBatchResults(data) {
  const resultsDiv = document.getElementById('batchResults');
  const countSpan = document.getElementById('batchRowCount');
  const timeSpan = document.getElementById('batchDuration');
  const downloadLink = document.getElementById('batchDownloadLink');
  const previewBody = document.getElementById('batchPreviewBody');

  if (!resultsDiv) return;

  if (countSpan) countSpan.textContent = data.total_rows;
  if (timeSpan) timeSpan.textContent = `${data.time_taken_seconds}s`;
  if (downloadLink) {
    downloadLink.href = data.download_url;
    downloadLink.style.display = 'inline-flex';
  }

  if (previewBody && data.preview) {
    previewBody.innerHTML = '';
    data.preview.forEach((row, i) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${escapeHtml(String(row.predicted_category || ''))}</td>
        <td>${escapeHtml(String(row.sentiment || ''))}</td>
        <td>${escapeHtml(String(row.emotion || ''))}</td>
        <td style="max-width:300px; font-size:0.85rem;">${escapeHtml(String(row.explanation || ''))}</td>
      `;
      previewBody.appendChild(tr);
    });
  }

  resultsDiv.style.display = 'block';
  resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// =====================================================================
// 4. HISTORY CONTROLLER
// =====================================================================
function initHistoryWorkspace() {
  const searchInput = document.getElementById('historySearchInput');
  const catFilter = document.getElementById('historyCategoryFilter');
  const sentFilter = document.getElementById('historySentimentFilter');
  const modeFilter = document.getElementById('historyModeFilter');
  const tbody = document.getElementById('historyTableBody');
  const paginationElem = document.getElementById('historyPagination');
  const clearBtn = document.getElementById('clearHistoryBtn');

  let currentPage = 1;

  async function loadHistory(page = 1) {
    currentPage = page;
    const category = catFilter ? catFilter.value : '';
    const sentiment = sentFilter ? sentFilter.value : '';
    const mode = modeFilter ? modeFilter.value : '';
    const search = searchInput ? searchInput.value.trim() : '';

    const params = new URLSearchParams({
      page: page,
      limit: 15,
      category: category,
      sentiment: sentiment,
      mode: mode,
      search: search
    });

    try {
      const res = await fetch(`/api/history?${params.toString()}`);
      const data = await res.json();
      renderHistoryTable(data.records);
      renderPagination(data.page, data.total_pages);
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  }

  function renderHistoryTable(records) {
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!records || records.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="empty-state">
            <p>No classification records found.</p>
          </td>
        </tr>
      `;
      return;
    }

    records.forEach(r => {
      const tr = document.createElement('tr');
      const confDisplay = r.confidence_score !== null && r.confidence_score !== undefined
        ? `${Math.round(r.confidence_score * 100)}%`
        : '<span style="color:var(--text-muted);">Rule Match</span>';

      tr.innerHTML = `
        <td><strong>#${r.id}</strong></td>
        <td style="max-width:280px; word-break:break-word;">${escapeHtml(r.input_text)}</td>
        <td><span class="badge badge-${r.predicted_category.toLowerCase().replace('/', '-')}">${r.predicted_category}</span></td>
        <td><span class="badge badge-sentiment-${(r.sentiment || 'neutral').toLowerCase()}">${r.sentiment || 'NEUTRAL'}</span></td>
        <td>${confDisplay}</td>
        <td style="font-size:0.8rem; color:var(--text-secondary);">${r.created_at || ''}</td>
        <td>
          <button class="btn btn-secondary btn-sm delete-btn" data-id="${r.id}" title="Delete record">
            &times;
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Attach delete listeners
    tbody.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const id = btn.getAttribute('data-id');
        if (confirm(`Delete record #${id}?`)) {
          await fetch(`/api/history/${id}`, { method: 'DELETE' });
          loadHistory(currentPage);
        }
      });
    });
  }

  function renderPagination(page, totalPages) {
    if (!paginationElem) return;
    paginationElem.innerHTML = '';
    if (totalPages <= 1) return;

    for (let p = 1; p <= totalPages; p++) {
      const btn = document.createElement('button');
      btn.className = `btn btn-sm ${p === page ? 'btn-primary' : 'btn-secondary'}`;
      btn.textContent = p;
      btn.style.margin = '0 3px';
      btn.addEventListener('click', () => loadHistory(p));
      paginationElem.appendChild(btn);
    }
  }

  if (searchInput) searchInput.addEventListener('input', () => loadHistory(1));
  if (catFilter) catFilter.addEventListener('change', () => loadHistory(1));
  if (sentFilter) sentFilter.addEventListener('change', () => loadHistory(1));
  if (modeFilter) modeFilter.addEventListener('change', () => loadHistory(1));

  if (clearBtn) {
    clearBtn.addEventListener('click', async () => {
      if (confirm('Are you sure you want to delete ALL classification history?')) {
        await fetch('/api/history/clear', { method: 'POST' });
        loadHistory(1);
      }
    });
  }

  if (tbody) {
    loadHistory(1);
  }
}

// =====================================================================
// 5. ANALYTICS CONTROLLER (Chart.js Integrations)
// =====================================================================
let categoryChart = null;
let sentimentChart = null;
let emotionChart = null;
let timelineChart = null;
let modeChart = null;

async function initAnalyticsDashboard() {
  const container = document.getElementById('analyticsDashboardView');
  if (!container) return;

  try {
    const res = await fetch('/api/analytics');
    const data = await res.json();
    renderAnalyticsCharts(data);
  } catch (err) {
    console.error('Failed to load analytics:', err);
  }
}

function renderAnalyticsCharts(data) {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const textColor = isDark ? '#9ca3af' : '#475569';
  const gridColor = isDark ? '#374151' : '#e2e8f0';

  // 1. Category Distribution Bar Chart
  const catCanvas = document.getElementById('categoryDistChart');
  if (catCanvas && data.category_distribution) {
    const labels = Object.keys(data.category_distribution);
    const counts = Object.values(data.category_distribution);

    if (categoryChart) categoryChart.destroy();
    categoryChart = new Chart(catCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Total Texts',
          data: counts,
          backgroundColor: '#3b82f6',
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: { beginAtZero: true, ticks: { color: textColor, precision: 0 }, grid: { color: gridColor } }
        }
      }
    });
  }

  // 2. Sentiment Pie/Doughnut Chart
  const sentCanvas = document.getElementById('sentimentDistChart');
  if (sentCanvas && data.sentiment_distribution) {
    const labels = ['POSITIVE', 'NEGATIVE', 'NEUTRAL'];
    const counts = labels.map(l => data.sentiment_distribution[l] || 0);

    if (sentimentChart) sentimentChart.destroy();
    sentimentChart = new Chart(sentCanvas.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: ['#10b981', '#ef4444', '#94a3b8']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: textColor } }
        }
      }
    });
  }

  // 3. Emotion Bar Chart
  const emoCanvas = document.getElementById('emotionDistChart');
  if (emoCanvas && data.emotion_distribution) {
    const labels = Object.keys(data.emotion_distribution);
    const counts = Object.values(data.emotion_distribution);

    if (emotionChart) emotionChart.destroy();
    emotionChart = new Chart(emoCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Occurrences',
          data: counts,
          backgroundColor: '#8b5cf6',
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: { beginAtZero: true, ticks: { color: textColor, precision: 0 }, grid: { color: gridColor } }
        }
      }
    });
  }

  // 4. Daily Activity Timeline
  const timeCanvas = document.getElementById('timelineDistChart');
  if (timeCanvas && data.daily_timeline) {
    const labels = data.daily_timeline.map(d => d.date);
    const counts = data.daily_timeline.map(d => d.count);

    if (timelineChart) timelineChart.destroy();
    timelineChart = new Chart(timeCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Classifications',
          data: counts,
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          fill: true,
          tension: 0.3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: textColor }, grid: { display: false } },
          y: { beginAtZero: true, ticks: { color: textColor, precision: 0 }, grid: { color: gridColor } }
        }
      }
    });
  }

  // 5. Mode Usage Chart
  const modeCanvas = document.getElementById('modeDistChart');
  if (modeCanvas && data.mode_distribution) {
    const labels = Object.keys(data.mode_distribution);
    const counts = Object.values(data.mode_distribution);

    if (modeChart) modeChart.destroy();
    modeChart = new Chart(modeCanvas.getContext('2d'), {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: counts,
          backgroundColor: ['#3b82f6', '#f59e0b', '#10b981', '#ec4899']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: textColor } }
        }
      }
    });
  }
}

// Model Retraining Button
function initModelRetrainBtn() {
  const btn = document.getElementById('retrainModelBtn');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Training ML Models...';
    try {
      const res = await fetch('/api/train', { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Training failed');
      alert(`Model retrained successfully! Best model: ${data.metrics.selected_model} (Accuracy: ${Math.round(data.metrics.selected_accuracy * 100)}%, F1: ${data.metrics.selected_f1_score})`);
      window.location.reload();
    } catch (err) {
      alert('Training Error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Retrain Model';
    }
  });
}

// Global DOM Loaded initialization
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initMobileSidebar();
  initClassifierWorkspace();
  initBatchWorkspace();
  initHistoryWorkspace();
  initAnalyticsDashboard();
  initModelRetrainBtn();
});
