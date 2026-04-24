const state = {
  configText: "",
  selectedRunPath: "",
  selectedJobId: "",
  selectedArtifactPath: "",
  selectedChartPath: "",
  followSelectedJob: false,
  quickDirty: false,
  editorDirty: false,
  lastConfigSource: "loaded",
};

const MODE_META = {
  main: {
    label: "研究检查",
    short: "先研究，再按配置决定是否继续仿真",
  },
  paper: {
    label: "仿真交易",
    short: "直接看账户权益、订单和成交结果",
  },
  live: {
    label: "实盘演练",
    short: "执行一轮 live dry-run，检查风控与适配器",
  },
};

const RUN_TYPE_LABELS = {
  research: "研究结果",
  main: "研究 + 单标的仿真",
  paper: "单标的仿真",
  portfolio_main: "研究 + 组合仿真",
  portfolio_paper: "组合仿真",
  portfolio_research: "组合研究",
  live: "实盘演练",
  unknown: "未识别结果",
};

const STATUS_LABELS = {
  queued: "排队中",
  running: "运行中",
  completed: "已完成",
  failed: "失败",
};

const MARKET_LABELS = {
  auto: "自动识别",
  us_equity: "美股",
  cn_equity: "A股",
  hk_equity: "港股",
  crypto_spot: "加密现货",
};

const SOURCE_LABELS = {
  api: "在线行情",
  csv: "本地 CSV",
};

const ADAPTER_LABELS = {
  paper_live: "模拟成交通道",
  real: "真实券商通道",
};

const METRIC_LABELS = {
  total_return: "总收益",
  annual_return: "年化收益",
  sharpe_ratio: "夏普比率",
  sortino_ratio: "索提诺比率",
  max_drawdown: "最大回撤",
  win_rate: "胜率",
  total_trades: "交易次数",
  recommendation_count: "建议条数",
  initial_equity: "初始权益",
  final_equity: "期末权益",
  orders: "订单数",
  fills: "成交笔数",
  closed_trades: "平仓笔数",
  submitted_orders: "已提交订单",
  rejected_orders: "被拒订单",
  canceled_orders: "已撤订单",
  realized_pnl: "已实现盈亏",
  unrealized_pnl: "浮动盈亏",
  fees_paid: "累计费用",
  rebalances: "调仓次数",
  active_symbols: "活跃标的数",
  symbol_count: "标的数量",
  completed_symbols: "完成标的数",
  processed_rows: "处理行数",
  benchmark_return: "基准收益",
  active_return: "相对基准",
  paper_total_return: "仿真收益",
  research_total_return: "研究收益",
  data_mode: "数据模式",
  config_fingerprint: "配置指纹",
  llm_used: "AI 复盘",
  session_blocks: "时段拦截次数",
  risk_blocks: "风控拦截次数",
  latest_timestamp: "最新时间",
};

const CONFIG_SYNC_TEXT = {
  clean: "常用参数和高级配置已经同步，可以直接运行。",
  quick: "你刚改了常用参数，启动前会自动写入高级配置。",
  editor: "你正在直接编辑高级配置，当前将以 YAML 为准。",
};

const elements = {
  healthText: document.getElementById("health-text"),
  configPath: document.getElementById("config-path"),
  resultsRoot: document.getElementById("results-root"),
  flashMessage: document.getElementById("flash-message"),
  configEditor: document.getElementById("config-editor"),
  configSyncStatus: document.getElementById("config-sync-status"),
  deskNote: document.getElementById("desk-note"),
  summarySymbol: document.getElementById("summary-symbol"),
  summaryMarket: document.getElementById("summary-market"),
  summaryPlan: document.getElementById("summary-plan"),
  summaryAdapter: document.getElementById("summary-adapter"),
  runsList: document.getElementById("runs-list"),
  jobsList: document.getElementById("jobs-list"),
  detailJson: document.getElementById("detail-json"),
  detailSummary: document.getElementById("detail-summary"),
  artifactsList: document.getElementById("artifacts-list"),
  saveBeforeRun: document.getElementById("save-before-run"),
  quickSymbol: document.getElementById("quick-symbol"),
  quickSymbols: document.getElementById("quick-symbols"),
  quickSource: document.getElementById("quick-source"),
  quickInterval: document.getElementById("quick-interval"),
  quickMarketProfile: document.getElementById("quick-market-profile"),
  quickLiveAdapter: document.getElementById("quick-live-adapter"),
  quickAllowShort: document.getElementById("quick-allow-short"),
  quickPaperEnabled: document.getElementById("quick-paper-enabled"),
  quickPortfolioEnabled: document.getElementById("quick-portfolio-enabled"),
  chartTabs: document.getElementById("chart-tabs"),
  chartMeta: document.getElementById("chart-meta"),
  chartCanvas: document.getElementById("chart-canvas"),
  jobLogOutput: document.getElementById("job-log-output"),
  previewPath: document.getElementById("preview-path"),
  filePreview: document.getElementById("file-preview"),
  aiReview: document.getElementById("ai-review"),
  aiReviewSource: document.getElementById("ai-review-source"),
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `请求失败: ${response.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function setFlash(message, kind = "info") {
  elements.flashMessage.textContent = message;
  elements.flashMessage.className = `inline-status badge--${kind}`;
}

function formatValue(value) {
  if (typeof value === "number") {
    if (Math.abs(value) < 1 && value !== 0) {
      return value.toFixed(4);
    }
    return value.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
  }
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  if (Array.isArray(value)) {
    return `${value.length} 项`;
  }
  if (typeof value === "object") {
    return "查看详情";
  }
  return String(value);
}

function formatMetric(value, key = "") {
  if (typeof value !== "number") {
    return formatValue(value);
  }
  const percentHints = ["return", "drawdown", "win_rate", "benchmark"];
  if (percentHints.some((item) => key.includes(item))) {
    return `${(value * 100).toFixed(2)}%`;
  }
  return formatValue(value);
}

function metricLabel(key) {
  return METRIC_LABELS[key] || String(key || "-").replace(/_/g, " ");
}

function modeLabel(mode) {
  return MODE_META[mode]?.label || String(mode || "未知任务");
}

function runTypeLabel(runType) {
  return RUN_TYPE_LABELS[runType] || String(runType || "未识别结果");
}

function statusLabel(status) {
  return STATUS_LABELS[status] || String(status || "未知状态");
}

function marketLabel(profile) {
  return MARKET_LABELS[profile] || profile || "自动识别";
}

function sourceLabel(source) {
  return SOURCE_LABELS[source] || source || "在线行情";
}

function adapterLabel(adapter) {
  return ADAPTER_LABELS[adapter] || adapter || "模拟成交通道";
}

function getSymbolsFromConfig(config) {
  const data = config?.data || {};
  const symbols = Array.isArray(data.symbols) ? data.symbols.map((item) => String(item).trim()).filter(Boolean) : [];
  const symbol = String(data.symbol || "").trim();
  if (symbols.length) {
    return symbols;
  }
  return symbol ? [symbol] : [];
}

function readQuickForm() {
  const typedSymbols = elements.quickSymbols.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const symbol = elements.quickSymbol.value.trim() || typedSymbols[0] || "aapl.us";
  return {
    data: {
      symbol,
      symbols: typedSymbols.length ? typedSymbols : [symbol],
      source: elements.quickSource.value,
      interval: elements.quickInterval.value.trim() || "d",
    },
    market: {
      profile: elements.quickMarketProfile.value,
    },
    strategy: {
      allow_short: elements.quickAllowShort.checked,
    },
    paper_trading: {
      enabled: elements.quickPaperEnabled.checked,
    },
    portfolio: {
      enabled: elements.quickPortfolioEnabled.checked,
    },
    live_trading: {
      adapter: elements.quickLiveAdapter.value,
    },
  };
}

function buildQuickPatch() {
  return readQuickForm();
}

function describePlan(config) {
  const symbols = getSymbolsFromConfig(config);
  const paperEnabled = Boolean(config?.paper_trading?.enabled);
  const portfolioEnabled = Boolean(config?.portfolio?.enabled) && symbols.length > 1;

  if (paperEnabled && portfolioEnabled) {
    return "研究 + 组合仿真";
  }
  if (paperEnabled) {
    return "研究 + 单标的仿真";
  }
  if (portfolioEnabled) {
    return "组合研究";
  }
  return "研究结果输出";
}

function updateWorkbenchSummary(config) {
  const symbols = getSymbolsFromConfig(config);
  const interval = config?.data?.interval || "d";
  const source = sourceLabel(config?.data?.source);
  const market = marketLabel(config?.market?.profile || "auto");
  const adapter = adapterLabel(config?.live_trading?.adapter || "paper_live");
  const symbolLabel = symbols.length > 1 ? `${symbols.length} 个标的` : (symbols[0] || "-" );
  const symbolDescription = symbols.length > 1 ? `${symbols[0]} 等 ${symbols.length} 个` : symbolLabel;
  const plan = describePlan(config);

  elements.summarySymbol.textContent = symbolLabel;
  elements.summaryMarket.textContent = market;
  elements.summaryPlan.textContent = plan;
  elements.summaryAdapter.textContent = adapter;
  elements.deskNote.textContent = `当前会用${source}加载 ${interval} 周期数据，按 ${market} 规则处理 ${symbolDescription}；右侧按钮决定你是只做研究、直接仿真，还是做一轮实盘演练。`;
}

function applyQuickConfig(config) {
  const data = config.data || {};
  const market = config.market || {};
  const strategy = config.strategy || {};
  const paperTrading = config.paper_trading || {};
  const portfolio = config.portfolio || {};
  const liveTrading = config.live_trading || {};

  elements.quickSymbol.value = data.symbol || "";
  elements.quickSymbols.value = Array.isArray(data.symbols) ? data.symbols.join(", ") : "";
  elements.quickSource.value = data.source || "api";
  elements.quickInterval.value = data.interval || "d";
  elements.quickMarketProfile.value = market.profile || "auto";
  elements.quickLiveAdapter.value = liveTrading.adapter || "paper_live";
  elements.quickAllowShort.checked = Boolean(strategy.allow_short);
  elements.quickPaperEnabled.checked = Boolean(paperTrading.enabled);
  elements.quickPortfolioEnabled.checked = Boolean(portfolio.enabled);
  updateWorkbenchSummary(config);
}

function setConfigSyncState(kind) {
  elements.configSyncStatus.className = `sync-banner sync-banner--${kind}`;
  elements.configSyncStatus.textContent = CONFIG_SYNC_TEXT[kind] || CONFIG_SYNC_TEXT.clean;
}

function markQuickDirty() {
  state.quickDirty = true;
  state.editorDirty = false;
  state.lastConfigSource = "quick";
  updateWorkbenchSummary(buildQuickPatch());
  setConfigSyncState("quick");
  setFlash("常用参数已更新，启动前会自动合并进高级配置", "running");
}

function markEditorDirty() {
  state.editorDirty = true;
  state.quickDirty = false;
  state.lastConfigSource = "editor";
  setConfigSyncState("editor");
  setFlash("高级配置已修改，当前以 YAML 为准", "running");
}

function setPreviewEmpty(message) {
  elements.previewPath.textContent = "-";
  elements.filePreview.className = "preview-shell empty-state";
  elements.filePreview.textContent = message;
}

function setChartEmpty(message) {
  elements.chartMeta.textContent = "选择一条结果后显示曲线";
  elements.chartCanvas.className = "chart-shell empty-state";
  elements.chartCanvas.textContent = message;
  elements.chartTabs.innerHTML = "";
}

function renderJobLogs(payload) {
  const logs = payload?.logs || [];
  elements.jobLogOutput.textContent = logs.length
    ? logs.join("\n")
    : "选择一条任务后，这里会显示该任务的最新日志。";
}

function canPreviewArtifact(name) {
  const lower = String(name || "").toLowerCase();
  return [".json", ".csv", ".txt", ".log", ".md", ".yaml", ".yml", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"].some((suffix) => lower.endsWith(suffix));
}

function pickDefaultArtifact(artifacts) {
  if (!artifacts.length) {
    return null;
  }
  const preferred = [
    "ai_review.json",
    "paper_summary.json",
    "portfolio_summary.json",
    "live_summary.json",
    "metrics.json",
    "paper_account_history.csv",
    "portfolio_account_history.csv",
    "live_account_history.csv",
    "equity_curve.csv",
    "portfolio_symbol_summary.json",
    "portfolio_symbol_history.csv",
    "paper_orders.csv",
  ];
  for (const name of preferred) {
    const matched = artifacts.find((artifact) => artifact.name === name);
    if (matched) {
      return matched;
    }
  }
  return artifacts.find((artifact) => canPreviewArtifact(artifact.name)) || artifacts[0];
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function extractTradingMetricEntries(payload) {
  const source = payload || {};
  const summary = source.summary || source.result?.summary || source || {};
  const metrics = source.metrics || source.result?.metrics || {};
  const performance = source.performance_metrics || source.performance || metrics.performance || summary.performance || {};
  const risk = source.risk_metrics || source.risk_metrics || source.risk || metrics.risk_metrics || summary.risk_metrics || {};
  const paper = source.paper_summary || source.paper_result_preview || source.result_preview || (
    summary.final_equity !== undefined || summary.orders !== undefined ? summary : {}
  );
  const aiReview = source.ai_review || source.result?.ai_review || metrics.ai_review || {};
  const runContext = source.run_context || source.result?.run_context || {};

  const entries = [];
  const paperReturn = paper.total_return;
  const researchReturn = performance.total_return;
  const benchmarkReturn = performance.benchmark_return;
  const activeReturn = typeof researchReturn === "number" && typeof benchmarkReturn === "number"
    ? researchReturn - benchmarkReturn
    : undefined;

  const add = (key, value) => {
    if (value !== undefined && value !== null && value !== "") {
      entries.push([key, value]);
    }
  };

  add("paper_total_return", paperReturn);
  add("research_total_return", researchReturn);
  add("benchmark_return", benchmarkReturn);
  add("active_return", activeReturn);
  add("sharpe_ratio", performance.sharpe_ratio);
  add("max_drawdown", firstDefined(risk.max_drawdown, paper.max_drawdown, summary.max_drawdown));
  add("win_rate", firstDefined(paper.win_rate, performance.win_rate, summary.win_rate));
  add("closed_trades", firstDefined(paper.closed_trades, performance.total_trades, summary.closed_trades));
  add("final_equity", paper.final_equity);
  add("llm_used", aiReview.llm_used);
  add("data_mode", runContext.data_mode);
  add("config_fingerprint", runContext.config_fingerprint);
  return entries;
}

function summarizeEntries(summary, limit = 3) {
  if (!summary || typeof summary !== "object") {
    return "等待完成";
  }
  const tradingEntries = extractTradingMetricEntries(summary).slice(0, limit);
  const entries = tradingEntries.length ? tradingEntries : Object.entries(summary).slice(0, limit);
  if (!entries.length) {
    return "暂无摘要";
  }
  return entries
    .map(([key, value]) => `${metricLabel(key)}: ${formatMetric(value, key)}`)
    .join(" · ");
}

function renderRunBadges(run) {
  const badges = [];
  const context = run.run_context || {};
  const review = run.ai_review || {};
  if (context.data_mode) {
    badges.push(context.data_mode);
  }
  if (context.config_fingerprint) {
    badges.push(`配置 ${context.config_fingerprint}`);
  }
  if (review.llm_used === true) {
    badges.push("LLM 复盘成功");
  } else if (review.llm_error) {
    badges.push("LLM 已降级");
  } else if (review.llm_used === false) {
    badges.push("本地复盘");
  }
  return badges.length
    ? `<div class="card-tags">${badges.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`
    : "";
}

async function loadHealth() {
  const payload = await fetchJson("/api/health");
  elements.healthText.textContent = `服务正常 · ${payload.server_time}`;
  elements.configPath.textContent = payload.config_path;
  elements.resultsRoot.textContent = payload.results_root;
}

async function loadConfig() {
  const payload = await fetchJson("/api/config");
  state.configText = payload.text;
  elements.configEditor.value = payload.text;
  applyQuickConfig(payload.config || {});
  state.quickDirty = false;
  state.editorDirty = false;
  state.lastConfigSource = "loaded";
  setConfigSyncState("clean");
}

async function syncQuickToYaml({ silent = false } = {}) {
  const payload = await fetchJson("/api/config/patch", {
    method: "POST",
    body: JSON.stringify({
      text: elements.configEditor.value,
      patch: buildQuickPatch(),
    }),
  });
  state.configText = payload.text;
  elements.configEditor.value = payload.text;
  applyQuickConfig(payload.config || {});
  state.quickDirty = false;
  state.editorDirty = false;
  state.lastConfigSource = "loaded";
  setConfigSyncState("clean");
  if (!silent) {
    setFlash("常用参数已写入高级配置", "ok");
  }
}

async function saveConfig() {
  if (state.quickDirty && state.lastConfigSource === "quick") {
    await syncQuickToYaml({ silent: true });
  }
  const payload = await fetchJson("/api/config", {
    method: "POST",
    body: JSON.stringify({ text: elements.configEditor.value }),
  });
  state.configText = payload.text;
  elements.configEditor.value = payload.text;
  applyQuickConfig(payload.config || {});
  state.quickDirty = false;
  state.editorDirty = false;
  state.lastConfigSource = "loaded";
  setConfigSyncState("clean");
  setFlash("配置已保存到磁盘", "ok");
}

function renderJobs(jobs) {
  if (!jobs.length) {
    elements.jobsList.className = "stack-list empty-state";
    elements.jobsList.textContent = "还没有启动任务";
    return;
  }

  elements.jobsList.className = "stack-list";
  elements.jobsList.innerHTML = jobs
    .map((job) => {
      const activeClass = state.selectedJobId === job.id ? "active" : "";
      const summary = job.result?.summary
        ? summarizeEntries(job.result.summary, 2)
        : (job.error?.message || "等待执行结果");
      const modeInfo = MODE_META[job.mode] || { label: job.mode, short: "" };
      return `
        <button class="job-card ${activeClass}" data-job-id="${escapeHtml(job.id)}">
          <div class="card-topline">
            <div>
              <strong>${escapeHtml(modeInfo.label)}</strong>
              <p class="card-kicker">${escapeHtml(modeInfo.short)}</p>
            </div>
            <span class="badge badge--${escapeHtml(job.status)}">${escapeHtml(statusLabel(job.status))}</span>
          </div>
          <div class="card-meta">创建于 ${escapeHtml(job.created_at)}</div>
          <div class="card-summary">${escapeHtml(summary)}</div>
          ${renderRunBadges(job.result || job)}
        </button>
      `;
    })
    .join("");

  elements.jobsList.querySelectorAll("[data-job-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedJobId = button.dataset.jobId;
      renderJobs(jobs);
      await loadJobLogs();
      const job = jobs.find((item) => item.id === state.selectedJobId);
      if (job?.result?.relative_output_dir) {
        await selectRun(job.result.relative_output_dir);
      } else if (job) {
        renderDetail(job);
        setChartEmpty("当前任务还没有生成结果曲线");
        setPreviewEmpty("任务尚未产生产物文件");
      }
    });
  });
}

function renderRuns(runs) {
  if (!runs.length) {
    elements.runsList.className = "stack-list empty-state";
    elements.runsList.textContent = "还没有检测到结果目录";
    return;
  }

  elements.runsList.className = "stack-list";
  elements.runsList.innerHTML = runs
    .map((run) => {
      const activeClass = state.selectedRunPath === run.relative_output_dir ? "active" : "";
      return `
        <button class="run-card ${activeClass}" data-run-path="${escapeHtml(run.relative_output_dir)}">
          <div class="card-topline">
            <div>
              <strong>${escapeHtml(runTypeLabel(run.run_type))}</strong>
              <p class="card-kicker">${escapeHtml(run.relative_output_dir)}</p>
            </div>
            <span class="badge badge--ok">${escapeHtml(String(run.artifact_count))} 个文件</span>
          </div>
          <div class="card-meta">更新于 ${escapeHtml(run.modified_at)}</div>
          <div class="card-summary">${escapeHtml(summarizeEntries(run, 4))}</div>
          ${renderRunBadges(run)}
        </button>
      `;
    })
    .join("");

  elements.runsList.querySelectorAll("[data-run-path]").forEach((button) => {
    button.addEventListener("click", async () => {
      await selectRun(button.dataset.runPath);
    });
  });
}

function renderAIReview(review) {
  if (!review) {
    elements.aiReviewSource.textContent = "-";
    elements.aiReview.className = "ai-review-shell empty-state";
    elements.aiReview.textContent = "本次结果还没有复盘报告";
    return;
  }

  const sourceText = review.llm_used ? "LLM 复盘" : "本地规则复盘";
  const errorBlock = review.llm_error
    ? `<div class="ai-review-warning">LLM 本次未完成：${escapeHtml(review.llm_error)}，已展示本地规则复盘。</div>`
    : "";
  const findings = Array.isArray(review.key_findings) ? review.key_findings : [];
  const risks = Array.isArray(review.risk_notes) ? review.risk_notes : [];
  const nextSteps = Array.isArray(review.next_steps) ? review.next_steps : [];
  const suggestions = Array.isArray(review.parameter_suggestions) ? review.parameter_suggestions : [];

  elements.aiReviewSource.textContent = sourceText;
  elements.aiReview.className = "ai-review-shell";
  elements.aiReview.innerHTML = `
    <div class="ai-review-header">
      <strong>${escapeHtml(review.headline || "复盘报告已生成")}</strong>
      <span class="ai-review-pill">${escapeHtml(sourceText)}</span>
    </div>
    ${errorBlock}
    <div class="review-columns">
      <section>
        <h4>关键发现</h4>
        <ul class="review-list">${findings.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>暂无关键发现</li>"}</ul>
      </section>
      <section>
        <h4>风险提示</h4>
        <ul class="review-list">${risks.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>暂无风险提示</li>"}</ul>
      </section>
    </div>
    <div class="suggestion-grid">
      ${suggestions.map((item) => `
        <article class="suggestion-card">
          <label>${escapeHtml(item.name || "参数")}</label>
          <strong>${escapeHtml(formatValue(item.current))} -> ${escapeHtml(formatValue(item.suggested))}</strong>
          <p>${escapeHtml(item.reason || "")}</p>
        </article>
      `).join("") || "<div class=\"empty-state\">暂无参数建议</div>"}
    </div>
    <section class="next-steps-block">
      <h4>下一步</h4>
      <ul class="review-list">${nextSteps.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>暂无下一步建议</li>"}</ul>
    </section>
  `;
}

function renderDetail(payload) {
  const summary = payload.summary || payload.result?.summary || {};
  const runContext = payload.run_context || payload.result?.run_context || {};
  const metricEntries = extractTradingMetricEntries(payload);
  const summaryEntries = metricEntries.length ? metricEntries : Object.entries(summary);
  if (!summaryEntries.length) {
    elements.detailSummary.className = "metric-grid empty-state";
    elements.detailSummary.textContent = "当前对象没有可显示的关键指标";
  } else {
    const repeatNote = runContext.repeat_note
      ? `<article class="metric-card metric-card--wide"><label>为什么多次结果一样</label><strong>${escapeHtml(runContext.repeat_note)}</strong></article>`
      : "";
    elements.detailSummary.className = "metric-grid";
    elements.detailSummary.innerHTML = summaryEntries
      .slice(0, 12)
      .map(([key, value]) => `
        <article class="metric-card">
          <label>${escapeHtml(metricLabel(key))}</label>
          <strong>${escapeHtml(formatMetric(value, key))}</strong>
        </article>
      `)
      .join("") + repeatNote;
  }

  elements.detailJson.textContent = JSON.stringify(payload, null, 2);
  renderAIReview(payload.ai_review || payload.result?.ai_review || payload.metrics?.ai_review);

  const artifacts = payload.artifacts || payload.result?.artifacts || [];
  if (!artifacts.length) {
    elements.artifactsList.className = "stack-list empty-state";
    elements.artifactsList.textContent = "当前结果目录没有产物文件";
    return;
  }

  elements.artifactsList.className = "stack-list";
  elements.artifactsList.innerHTML = artifacts
    .map((artifact) => {
      const activeClass = state.selectedArtifactPath === artifact.relative_path ? "active" : "";
      const previewButton = canPreviewArtifact(artifact.name)
        ? `<button class="ghost-button ghost-button--small" data-preview-path="${escapeHtml(artifact.relative_path)}">页内预览</button>`
        : `<span class="artifact-hint">仅支持单独打开</span>`;
      return `
        <article class="artifact-card ${activeClass}">
          <div class="artifact-headline">
            <strong>${escapeHtml(artifact.name)}</strong>
            <span class="badge badge--ok">${escapeHtml(formatValue(artifact.size_bytes))} B</span>
          </div>
          <small>${escapeHtml(artifact.relative_path)}</small>
          <small>${escapeHtml(artifact.modified_at || "")}</small>
          <div class="artifact-actions">
            ${previewButton}
            <a href="/files/${encodeURIComponent(artifact.relative_path)}" target="_blank" rel="noreferrer">单独打开</a>
          </div>
        </article>
      `;
    })
    .join("");

  elements.artifactsList.querySelectorAll("[data-preview-path]").forEach((button) => {
    button.addEventListener("click", async () => {
      await previewFile(button.dataset.previewPath);
    });
  });
}

function buildChartSvg(chart) {
  const allPoints = chart.series.flatMap((series) => series.points || []);
  if (!allPoints.length) {
    return "";
  }

  const width = 960;
  const height = 300;
  const padding = { top: 24, right: 20, bottom: 36, left: 56 };
  const palette = ["#1c5a46", "#a25a23", "#456f8f", "#8e3f4a"];
  const usableWidth = width - padding.left - padding.right;
  const usableHeight = height - padding.top - padding.bottom;
  const allValues = allPoints.map((point) => Number(point.y));
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = maxValue - minValue || Math.max(Math.abs(maxValue), 1);
  const lower = minValue - range * 0.08;
  const upper = maxValue + range * 0.08;

  const gridLines = [0, 0.25, 0.5, 0.75, 1].map((tick) => {
    const y = padding.top + usableHeight * tick;
    const value = upper - (upper - lower) * tick;
    return `
      <g>
        <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="chart-grid" />
        <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" class="chart-axis-label">${escapeHtml(formatValue(value))}</text>
      </g>
    `;
  }).join("");

  const polylines = chart.series.map((series, seriesIndex) => {
    const points = series.points || [];
    if (!points.length) {
      return "";
    }
    const denominator = Math.max(points.length - 1, 1);
    const coordinates = points.map((point, index) => {
      const x = padding.left + (usableWidth * (points.length === 1 ? 0.5 : index / denominator));
      const y = padding.top + ((upper - Number(point.y)) / (upper - lower)) * usableHeight;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    return `<polyline fill="none" stroke="${palette[seriesIndex % palette.length]}" stroke-width="3" points="${coordinates}" />`;
  }).join("");

  const firstSeries = chart.series.find((series) => (series.points || []).length) || chart.series[0];
  const xLabels = firstSeries?.points || [];
  const labelIndexes = [0, Math.floor((xLabels.length - 1) / 2), xLabels.length - 1]
    .filter((value, index, array) => value >= 0 && array.indexOf(value) === index);
  const bottomLabels = labelIndexes.map((index) => {
    const denominator = Math.max(xLabels.length - 1, 1);
    const x = padding.left + (usableWidth * (xLabels.length === 1 ? 0.5 : index / denominator));
    return `<text x="${x}" y="${height - 8}" text-anchor="middle" class="chart-axis-label">${escapeHtml(xLabels[index].x)}</text>`;
  }).join("");

  const legend = chart.series.map((series, index) => `
    <span class="legend-pill">
      <i style="background:${palette[index % palette.length]}"></i>
      ${escapeHtml(series.label)}
    </span>
  `).join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" class="chart-svg" role="img" aria-label="${escapeHtml(chart.title)}">
      ${gridLines}
      ${polylines}
      ${bottomLabels}
    </svg>
    <div class="chart-legend">${legend}</div>
  `;
}

function renderCharts(payload) {
  const charts = payload?.charts || [];
  if (!charts.length) {
    setChartEmpty("当前结果目录没有可绘制的 CSV 曲线");
    return;
  }

  if (!charts.some((chart) => chart.relative_path === state.selectedChartPath)) {
    state.selectedChartPath = charts[0].relative_path;
  }

  elements.chartTabs.innerHTML = charts
    .map((chart) => {
      const activeClass = chart.relative_path === state.selectedChartPath ? "active" : "";
      return `<button class="chart-tab ${activeClass}" data-chart-path="${escapeHtml(chart.relative_path)}">${escapeHtml(chart.file_name)}</button>`;
    })
    .join("");

  elements.chartTabs.querySelectorAll("[data-chart-path]").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedChartPath = button.dataset.chartPath;
      renderCharts(payload);
    });
  });

  const activeChart = charts.find((chart) => chart.relative_path === state.selectedChartPath) || charts[0];
  elements.chartMeta.textContent = `${activeChart.title} · ${activeChart.point_count || 0} 个点 · ${activeChart.x_start || "-"} -> ${activeChart.x_end || "-"}`;
  if (!activeChart.series?.length) {
    elements.chartCanvas.className = "chart-shell empty-state";
    elements.chartCanvas.textContent = "当前曲线文件没有可绘制的数值列";
    return;
  }

  elements.chartCanvas.className = "chart-shell";
  elements.chartCanvas.innerHTML = buildChartSvg(activeChart);
}

function renderTablePreview(payload) {
  const columns = payload.columns || [];
  const rows = payload.rows || [];
  const head = columns.map((column) => `<th>${escapeHtml(metricLabel(column))}</th>`).join("");
  const body = rows.length
    ? rows.map((row) => `
      <tr>
        ${columns.map((column) => `<td>${escapeHtml(formatValue(row[column]))}</td>`).join("")}
      </tr>
    `).join("")
    : `<tr><td colspan="${Math.max(columns.length, 1)}">空表</td></tr>`;

  const meta = payload.truncated ? `已显示前 ${rows.length} 行，共 ${payload.row_count} 行` : `共 ${payload.row_count || rows.length} 行`;
  elements.filePreview.className = "preview-shell";
  elements.filePreview.innerHTML = `
    <div class="preview-meta">${escapeHtml(meta)}</div>
    <div class="table-wrap">
      <table class="preview-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function renderFilePreview(payload) {
  elements.previewPath.textContent = payload.path || "-";

  if (payload.kind === "table") {
    renderTablePreview(payload);
    return;
  }

  if (payload.kind === "json") {
    elements.filePreview.className = "preview-shell";
    elements.filePreview.innerHTML = `<pre class="code-block code-block--compact">${escapeHtml(payload.text || JSON.stringify(payload.content || {}, null, 2))}</pre>`;
    return;
  }

  if (payload.kind === "text") {
    elements.filePreview.className = "preview-shell";
    elements.filePreview.innerHTML = `<pre class="code-block code-block--compact">${escapeHtml(payload.text || "")}</pre>`;
    return;
  }

  if (payload.kind === "image") {
    elements.filePreview.className = "preview-shell";
    elements.filePreview.innerHTML = `<img class="preview-image" src="${escapeHtml(payload.file_url)}" alt="${escapeHtml(payload.name)}">`;
    return;
  }

  elements.filePreview.className = "preview-shell empty-state";
  elements.filePreview.textContent = payload.message || "当前文件不支持页内预览";
}

async function loadJobLogs() {
  if (!state.selectedJobId) {
    renderJobLogs();
    return;
  }
  const payload = await fetchJson(`/api/jobs/${encodeURIComponent(state.selectedJobId)}/logs?limit=160`);
  renderJobLogs(payload);
}

async function loadJobs(autoSelect = false) {
  const payload = await fetchJson("/api/jobs");
  const jobs = payload.jobs || [];

  if (autoSelect && !jobs.some((job) => job.id === state.selectedJobId) && jobs.length) {
    state.selectedJobId = jobs[0].id;
  }

  renderJobs(jobs);

  if (state.followSelectedJob && state.selectedJobId) {
    const selectedJob = jobs.find((job) => job.id === state.selectedJobId);
    if (selectedJob?.result?.relative_output_dir) {
      state.followSelectedJob = false;
      await selectRun(selectedJob.result.relative_output_dir);
    }
  }

  return jobs;
}

async function loadRuns() {
  const payload = await fetchJson("/api/results?limit=30");
  const runs = payload.runs || [];
  renderRuns(runs);
  return runs;
}

async function previewFile(relativePath, quiet = false) {
  if (!relativePath) {
    setPreviewEmpty("未选择要预览的文件");
    return;
  }

  try {
    state.selectedArtifactPath = relativePath;
    const payload = await fetchJson(`/api/file-preview?path=${encodeURIComponent(relativePath)}&limit=30`);
    renderFilePreview(payload);
  } catch (error) {
    if (!quiet) {
      setFlash(error.message, "error");
    }
    setPreviewEmpty(error.message);
  }
}

async function selectRun(relativePath) {
  state.selectedRunPath = relativePath;
  const [detailPayload, chartPayload] = await Promise.all([
    fetchJson(`/api/result?path=${encodeURIComponent(relativePath)}`),
    fetchJson(`/api/result/chart?path=${encodeURIComponent(relativePath)}`),
  ]);
  renderDetail(detailPayload);
  renderCharts(chartPayload);

  const artifacts = detailPayload.artifacts || [];
  const selectedArtifact = artifacts.find((artifact) => artifact.relative_path === state.selectedArtifactPath);
  const defaultArtifact = selectedArtifact || pickDefaultArtifact(artifacts);
  if (defaultArtifact && canPreviewArtifact(defaultArtifact.name)) {
    await previewFile(defaultArtifact.relative_path, true);
  } else {
    setPreviewEmpty("当前结果没有可直接预览的重点文件");
  }
  await loadRuns();
}

async function startJob(mode) {
  const modeInfo = MODE_META[mode] || { label: mode };
  if (state.quickDirty && state.lastConfigSource === "quick") {
    await syncQuickToYaml({ silent: true });
  }

  setFlash(`正在启动${modeInfo.label}...`, "running");
  const payload = await fetchJson("/api/jobs", {
    method: "POST",
    body: JSON.stringify({
      mode,
      config_text: elements.configEditor.value,
      save_config: elements.saveBeforeRun.checked,
    }),
  });
  state.selectedJobId = payload.id;
  state.followSelectedJob = true;
  renderJobLogs({ logs: [`任务已提交：${modeInfo.label}，等待执行日志...`] });
  setFlash(`任务已启动：${modeInfo.label} · ${payload.id}`, "running");
  await loadJobs();
  await loadJobLogs();
}

function bindQuickFieldEvents() {
  [elements.quickSymbol, elements.quickSymbols, elements.quickInterval].forEach((input) => {
    input.addEventListener("input", markQuickDirty);
  });
  [
    elements.quickSource,
    elements.quickMarketProfile,
    elements.quickLiveAdapter,
    elements.quickAllowShort,
    elements.quickPaperEnabled,
    elements.quickPortfolioEnabled,
  ].forEach((element) => {
    element.addEventListener("change", markQuickDirty);
  });
}

async function bootstrap() {
  try {
    const [jobs, runs] = await Promise.all([
      (async () => {
        await loadHealth();
        await loadConfig();
        return loadJobs(true);
      })(),
      loadRuns(),
    ]);

    if (state.selectedJobId && jobs.length) {
      await loadJobLogs();
    } else {
      renderJobLogs();
    }

    if (runs.length) {
      await selectRun(runs[0].relative_output_dir);
    } else {
      setChartEmpty("还没有结果目录，先从右侧启动一条任务");
      setPreviewEmpty("选中结果后，这里会自动打开最关键的文件");
    }

    setFlash("工作台已就绪，可以直接运行任务或查看最近一次结果", "ok");
  } catch (error) {
    console.error(error);
    setFlash(error.message, "error");
  }
}

setInterval(async () => {
  try {
    await Promise.all([loadJobs(), loadRuns()]);
    if (state.selectedJobId) {
      await loadJobLogs();
    }
  } catch (error) {
    console.error(error);
  }
}, 4000);

document.getElementById("sync-quick").addEventListener("click", () => syncQuickToYaml());
document.getElementById("save-config").addEventListener("click", saveConfig);
document.getElementById("reload-config").addEventListener("click", loadConfig);
document.getElementById("refresh-jobs").addEventListener("click", loadJobs);
document.getElementById("refresh-results").addEventListener("click", loadRuns);
document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => startJob(button.dataset.mode));
});
elements.configEditor.addEventListener("input", markEditorDirty);

bindQuickFieldEvents();
bootstrap();
