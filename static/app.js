const state = {
  cameraDashboard: null,
  alerts: [],
  fallEvents: [],
  reviewAlerts: [],
  showcaseCases: [],
  evaluation: null,
  selectedAlertId: null,
  selectedFallEventId: null,
  selectedReviewAlertId: null,
  selectedShowcaseId: null,
  activeAlertFilter: "all",
  activeShowcaseFilter: "all",
  eventPages: {
    alerts: 0,
    fallEvents: 0,
    reviewAlerts: 0,
    showcase: 0,
  },
  visibleLimits: {
    latestAlerts: 8,
    alerts: 8,
    fallEvents: 8,
    reviewAlerts: 8,
    showcase: 8,
  },
  reminderPollingStarted: false,
  reminders: [],
};

const labels = {
  high_risk: "高风险摔倒告警",
  low_risk: "低风险摔倒告警",
  handled: "已处理",
  no_alarm: "无护工告警",
  pending_detection: "检测处理中",
  normal: "正常",
};

const pages = [
  "intro",
  "monitoring",
  "alerts",
  "fall-events",
  "review-alerts",
  "showcase-evaluation",
  "architecture",
];

const EVENT_PAGE_SIZE = 4;

document.addEventListener("DOMContentLoaded", () => {
  renderStaticShell();
  setActivePage(getPageFromHash());
  window.addEventListener("hashchange", () => {
    setActivePage(getPageFromHash());
  });
  loadDynamicData();
});

async function loadDynamicData() {
  try {
    const [
      cameras,
      alertPayload,
      fallPayload,
      reviewPayload,
      showcasePayload,
      evaluation,
    ] = await Promise.all([
      fetchJson("/api/cameras"),
      fetchJson("/api/alerts"),
      fetchJson("/api/fall-events"),
      fetchJson("/api/review-alerts"),
      fetchJson("/api/showcase"),
      fetchJson("/api/evaluation"),
    ]);

    state.cameraDashboard = cameras;
    state.alerts = alertPayload.alerts || [];
    state.fallEvents = fallPayload.events || [];
    state.reviewAlerts = reviewPayload.alerts || [];
    state.showcaseCases = showcasePayload.cases || [];
    state.evaluation = evaluation;
    state.selectedAlertId = state.alerts[0]?.event_id || null;
    state.selectedFallEventId = state.fallEvents[0]?.event_id || null;
    state.selectedReviewAlertId = state.reviewAlerts[0]?.event_id || null;
    state.selectedShowcaseId = state.showcaseCases[0]?.event_id || null;

    renderMonitoring();
    renderAlertsCenter();
    renderFallEvents();
    renderReviewAlerts();
    renderShowcaseEvaluation();
    startReminderPolling();
  } catch (error) {
    document.querySelectorAll(".dynamic-target").forEach((node) => {
      node.innerHTML = `<p class="error">数据加载失败：${escapeHtml(error.message)}</p>`;
    });
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `${url} 请求失败`);
  }
  return payload;
}

function renderStaticShell() {
  renderIntro();
  renderMonitoringLoading();
  renderAlertsCenterLoading();
  renderFallEventsLoading();
  renderReviewAlertsLoading();
  renderShowcaseEvaluationLoading();
  renderArchitecture();
}

function renderIntro() {
  const section = document.querySelector("#intro");
  section.innerHTML = `
    <div class="hero-layout duty-hero">
      <div class="hero-copy">
        <p class="eyebrow">摄像头风险监测原型</p>
        <h2>面向养老照护现场的摄像头摔倒风险监测平台</h2>
        <p>
          把照护现场的摄像头画面组织成值班态势：检测模块输出风险等级后立即告警，
          护工在普通页面只看到告警状态和隐私预览占位。
        </p>
        <div class="hero-status-strip" aria-label="系统能力边界">
          <span>YOLO 高召回</span>
          <span>VLM 风险分级</span>
          <span>原始视频隔离</span>
        </div>
        <div class="hero-actions">
          <a class="button-link" href="#page=monitoring">进入实时监测</a>
          <a class="button-link secondary" href="#page=architecture">查看技术方案</a>
        </div>
      </div>
      <aside class="surveillance-wall hero-wall" aria-label="照护值班监控墙示意">
        <div class="wall-title">
          <span>照护值班态势</span>
          <strong>4 区域</strong>
        </div>
        <div class="wall-grid">
          <div class="wall-tile is-normal">
            <span>走廊 A</span>
            <strong>正常</strong>
          </div>
          <div class="wall-tile is-review">
            <span>活动室 B</span>
            <strong>低风险</strong>
          </div>
          <div class="wall-tile is-alert">
            <span>卧室 C</span>
            <strong>摔倒告警</strong>
          </div>
          <div class="wall-tile is-normal">
            <span>护理站 D</span>
            <strong>正常</strong>
          </div>
        </div>
        <div class="hero-rail">
          ${renderEvidenceRail({
            display_status: "high_risk",
            yolo_score: 0.84,
            vlm_label: "高风险摔倒告警",
          })}
        </div>
      </aside>
    </div>
    <div class="capability-grid">
      ${[
        ["摄像头风险监测", "以摄像头和区域为主语组织事件，而不是把结果呈现成视频文件列表。"],
        ["摔倒告警生成", "YOLO 与 VLM 作为检测模块输出高风险或低风险告警。"],
        ["告警优先处置", "高风险和低风险都立即进入护工处置队列，区别只在优先级和重复提醒频率。"],
        ["剪影隐私预览", "普通告警页只播放受控剪影预览，生成中或失败时只显示状态，不回退播放原始事件片段。"],
        ["处置闭环", "护工点击已处理后，事件变为已处理状态并停止重复提醒。"],
        ["后续证据能力", "加密和证据调阅将在后续阶段接入，不属于当前实现范围。"],
      ]
        .map(
          ([title, body]) => `
            <article class="capability-card">
              <h3>${title}</h3>
              <p>${body}</p>
            </article>
          `,
        )
        .join("")}
    </div>
    <div class="grid-2" style="margin-top: 16px;">
      <article class="panel">
        <h3>场景痛点</h3>
        <ul class="list-plain">
          <li>摔倒发现不及时，照护响应依赖人工巡查。</li>
          <li>普通监控需要事后查找，无法主动形成风险告警。</li>
          <li>照护视频隐私敏感，后续需要受控查看。</li>
          <li>事件证据需要更可信的留存和完整性说明。</li>
        </ul>
      </article>
      <article class="panel">
        <h3>当前实现状态</h3>
        <ul class="list-plain">
          <li>已实现摄像头风险告警、YOLO 候选检测、事件帧缓存和内部片段保存。</li>
          <li>已实现异步 VLM 风险分级、SQLite 告警状态记录和重复提醒。</li>
          <li>剪影预览、隐私加密、哈希校验、防删除证据留存和授权查看属于后续方向。</li>
        </ul>
      </article>
    </div>
  `;
}

function renderMonitoringLoading() {
  document.querySelector("#monitoring").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Monitoring workspace</p>
      <h2>实时监测</h2>
      <p>摄像头监控工作台正在加载事件数据。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderMonitoring() {
  const dashboard = state.cameraDashboard || {};
  const summary = dashboard.summary || {};
  const section = document.querySelector("#monitoring");
  section.innerHTML = `
    <div class="section-header dashboard-header">
      <div>
        <p class="eyebrow">Monitoring workspace</p>
        <h2>实时监测</h2>
        <p>按摄像头聚合当前事件数据，先看高风险区域，再进入告警处置。</p>
      </div>
      <div class="desk-clock">
        <span>值班视图</span>
        <strong>只读演示</strong>
      </div>
    </div>
    <div class="status-strip">
      ${metricCard("摄像头数量", summary.camera_count)}
      ${metricCard("风险摄像头", summary.risk_camera_count)}
      ${metricCard("高风险告警", summary.high_risk)}
      ${metricCard("低风险告警", summary.low_risk)}
      ${metricCard("已处理", summary.handled)}
      ${metricCard("无护工告警", summary.no_alarm)}
    </div>
    ${renderReminderBanner()}
    <div class="workspace-layout">
      <div>
        ${renderCameraCards(dashboard.cameras || [])}
      </div>
      <aside class="side-panel">
        <div class="panel-heading">
          <span>告警队列</span>
          <strong>最新优先</strong>
        </div>
        ${renderLatestAlerts(dashboard.latest_alerts || [])}
        <div class="note">
          ${(dashboard.simulation_note || "").trim()}
        </div>
        ${renderQueueSummary(dashboard.queue || {}, summary)}
      </aside>
    </div>
  `;
}

function renderCameraCards(cameras) {
  if (!cameras.length) {
    return `<div class="empty-state">暂无摄像头事件数据</div>`;
  }
  return `
    <div class="monitor-grid surveillance-wall">
      ${cameras
        .map(
          (camera) => `
            <article class="camera-card ${escapeAttr(camera.risk_status || "normal")}">
              <div class="camera-card-header">
                <div>
                  <h3>${escapeHtml(camera.camera_id)}</h3>
                  <p>${escapeHtml(camera.area_label || "模拟区域")} · ${escapeHtml(camera.online_status || "模拟在线")}</p>
                </div>
                ${badge(camera.risk_status)}
              </div>
              <div class="camera-card-body">
                <div class="camera-placeholder">
                  <span class="camera-corner top-left"></span>
                  <span class="camera-corner top-right"></span>
                  <span class="camera-corner bottom-left"></span>
                  <span class="camera-corner bottom-right"></span>
                  <div class="camera-feed-label">${escapeHtml(camera.placeholder_text || "")}</div>
                </div>
              </div>
              <div class="camera-card-footer">
                <div class="data-pair">
                  <span>最近告警时间</span>
                  <strong>${escapeHtml(camera.last_alert_time || "暂无")}</strong>
                </div>
                <div class="data-pair">
                  <span>最近告警类型</span>
                  <strong>${escapeHtml(camera.last_alert_type || "暂无")}</strong>
                </div>
                <div class="data-pair">
                  <span>低风险待处理</span>
                  <strong>${formatNumber(camera.pending_review_count)}</strong>
                </div>
                <div class="data-pair">
                  <span>高风险告警</span>
                  <strong>${formatNumber(camera.confirmed_event_count)}</strong>
                </div>
              </div>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderLatestAlerts(alerts) {
  if (!alerts.length) {
    return `<div class="empty-state">当前暂无告警。</div>`;
  }
  const visible = limitedEvents(alerts, "latestAlerts");
  return `
    <div class="alert-list">
      ${visible
        .map(
          (event) => `
            <button class="alert-item" type="button" onclick="openAlertFromQueue('${escapeAttr(event.event_id)}')">
              <div class="row-top">
                <span class="row-title">${escapeHtml(event.camera_id || "未知摄像头")}</span>
                ${badge(event.display_status)}
              </div>
              <div class="row-meta">${escapeHtml(event.created_at || "时间未记录")}</div>
              ${renderEvidenceRail(event)}
            </button>
          `,
        )
        .join("")}
    </div>
    ${renderShowMore("latestAlerts", alerts.length, "renderMonitoring")}
  `;
}

function renderQueueSummary(queue, summary) {
  const jobs = queue.jobs || {};
  const pending = jobs.pending ?? 0;
  return `
    <div class="panel" style="margin-top: 14px;">
      <h3>待处理摘要</h3>
      <div class="detail-grid">
        <div class="detail-item"><span>待 VLM 分级</span><strong>${formatNumber(pending)}</strong></div>
        <div class="detail-item"><span>低风险待处理</span><strong>${formatNumber(summary.low_risk)}</strong></div>
        <div class="detail-item"><span>队列状态</span><strong>${queue.available ? "可读取" : "未读取"}</strong></div>
        <div class="detail-item"><span>高风险摄像头</span><strong>${summary.risk_camera_count ? "存在" : "暂无"}</strong></div>
      </div>
    </div>
  `;
}

function renderAlertsCenterLoading() {
  document.querySelector("#alerts").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Alert center</p>
      <h2>告警中心</h2>
      <p>正在加载高风险和低风险护工告警。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderAlertsCenter() {
  const filters = [
    ["all", "全部告警"],
    ["high_risk", "高风险"],
    ["low_risk", "低风险"],
    ["handled", "已处理"],
  ];
  const events = filteredAlerts();
  document.querySelector("#alerts").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Alert center</p>
      <h2>告警中心</h2>
      <p>展示检测模块输出的高风险和低风险护工告警。</p>
    </div>
    <div class="filter-bar">
      ${filters
        .map(
          ([value, label]) => `
            <button type="button" class="${state.activeAlertFilter === value ? "active" : ""}" onclick="setAlertFilter('${value}')">${label}</button>
          `,
        )
        .join("")}
    </div>
    <div class="event-layout">
      <div>${renderEventTable(events, state.selectedAlertId, "selectAlert", "alerts")}</div>
      <div class="detail-panel sticky-detail" id="alert-detail"></div>
    </div>
  `;
  renderEventDetail("#alert-detail", state.selectedAlertId);
}

function renderFallEventsLoading() {
  document.querySelector("#fall-events").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Confirmed events</p>
      <h2>高风险摔倒告警</h2>
      <p>正在加载高风险摔倒告警。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderFallEvents() {
  document.querySelector("#fall-events").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Confirmed events</p>
      <h2>高风险摔倒告警</h2>
      <p>只展示高优先级护工告警，红色提醒默认每 20 秒重复触发。</p>
    </div>
    <div class="event-layout">
      <div>${renderEventTable(state.fallEvents, state.selectedFallEventId, "selectFallEvent", "fallEvents")}</div>
      <div class="detail-panel sticky-detail" id="fall-event-detail"></div>
    </div>
    <div class="note">普通告警页不展示原始视频。隐私预览生成中会显示状态，生成后只播放剪影预览，失败时提示到场核验。</div>
  `;
  renderEventDetail("#fall-event-detail", state.selectedFallEventId);
}

function renderReviewAlertsLoading() {
  document.querySelector("#review-alerts").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Review queue</p>
      <h2>低风险摔倒告警</h2>
      <p>正在加载低风险摔倒告警。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderReviewAlerts() {
  document.querySelector("#review-alerts").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Review queue</p>
      <h2>低风险摔倒告警</h2>
      <p>展示低优先级护工告警，黄色提醒默认每 60 秒重复触发。</p>
    </div>
    <div class="event-layout">
      <div>${renderEventTable(state.reviewAlerts, state.selectedReviewAlertId, "selectReviewAlert", "reviewAlerts")}</div>
      <div class="detail-panel sticky-detail" id="review-alert-detail"></div>
    </div>
    <div class="note">低风险告警同样需要立即到场核验；“已处理”只表示护工已接警处置。</div>
  `;
  renderEventDetail("#review-alert-detail", state.selectedReviewAlertId);
}

function renderShowcaseEvaluationLoading() {
  document.querySelector("#showcase-evaluation").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Showcase and evaluation</p>
      <h2>案例回放与评估</h2>
      <p>正在加载可展示案例和评估摘要。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderShowcaseEvaluation() {
  const filters = [
    ["all", "全部可展示案例"],
    ["high_risk", "高风险"],
    ["low_risk", "低风险"],
    ["handled", "已处理"],
  ];
  const cases = filteredShowcaseCases();
  document.querySelector("#showcase-evaluation").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Showcase and evaluation</p>
      <h2>告警案例与评估</h2>
      <p>案例页只展示普通告警数据，不提供原始视频调阅。</p>
    </div>
    <div class="grid-2">
      <div>
        <div class="filter-bar">
          ${filters
            .map(
              ([value, label]) => `
                <button type="button" class="${state.activeShowcaseFilter === value ? "active" : ""}" onclick="setShowcaseFilter('${value}')">${label}</button>
              `,
            )
            .join("")}
        </div>
        ${renderEventTable(cases, state.selectedShowcaseId, "selectShowcase", "showcase")}
      </div>
      <div>
        ${renderEvaluationCards(state.evaluation || {})}
      </div>
    </div>
    <div class="detail-panel" id="showcase-detail" style="margin-top: 18px;"></div>
  `;
  renderEventDetail("#showcase-detail", state.selectedShowcaseId);
}

function renderEventTable(events, selectedId, onSelectName, limitKey) {
  if (!events.length) {
    return `<div class="empty-state">当前暂无可展示记录。</div>`;
  }
  const visible = pagedEvents(events, limitKey);
  return `
    <div class="paginated-list">
      <div class="event-list">
        ${visible
          .map(
            (event) => `
              <button class="event-row ${event.event_id === selectedId ? "active" : ""}" type="button" onclick="${onSelectName}('${escapeAttr(event.event_id)}')">
                <div class="row-top">
                  <span class="row-title">${escapeHtml(event.event_id)}</span>
                  ${badge(event.display_status)}
                </div>
                <div class="row-meta">${escapeHtml(event.camera_id || "未知摄像头")} · ${escapeHtml(event.area_label || "模拟区域")}</div>
                <div class="row-meta">${escapeHtml(event.created_at || "时间未记录")}</div>
                ${renderEvidenceRail(event)}
              </button>
            `,
          )
          .join("")}
      </div>
      ${renderEventPagination(limitKey, events.length)}
    </div>
  `;
}

function renderReminderBanner() {
  if (!state.reminders.length) {
    return "";
  }
  const latest = state.reminders.slice(0, 3);
  return `
    <div class="reminder-banner" aria-live="polite">
      <div>
        <span>重复提醒</span>
        <strong>${latest.length} 条告警到期</strong>
      </div>
      <div class="reminder-list">
        ${latest
          .map(
            (reminder) => `
              <button type="button" class="reminder-chip ${escapeAttr(reminder.risk_level)}" onclick="openAlertFromQueue('${escapeAttr(reminder.event_id)}')">
                ${escapeHtml(labels[reminder.risk_level] || reminder.risk_level)}
                · ${escapeHtml(reminder.camera_id || "未知摄像头")}
                · 第 ${formatNumber(reminder.reminder_count)} 次
              </button>
            `,
          )
          .join("")}
      </div>
    </div>
  `;
}

async function renderEventDetail(containerSelector, eventId) {
  const container = document.querySelector(containerSelector);
  if (!container) return;
  if (!eventId) {
    container.innerHTML = `<div class="empty-state">请选择一条记录查看回放详情。</div>`;
    return;
  }
  container.innerHTML = `<p class="loading">详情加载中...</p>`;
  try {
    const detail = await fetchJson(`/api/events/${encodeURIComponent(eventId)}`);
    const event = detail.event || {};
    const candidate = detail.candidate || {};
    const verification = detail.verification || {};
    const status = detail.status_explanations || {};
    const displayStatus = event.display_status || event.risk_level || "normal";
    const currentDecision =
      event.vlm_label || event.status_label || labels[displayStatus] || labels[event.risk_level] || displayStatus;
    const vlmConfidence =
      verification.confidence ?? event.vlm_confidence ?? "未记录";
    const evidenceStatus =
      privacyPreviewStatusLabel(event.privacy_preview_status);
    const handleButton =
      event.can_handle === true && event.alert_status === "pending"
        ? `<button class="button-link handle-button" type="button" onclick="handleEvent('${escapeAttr(event.event_id)}')">已处理</button>`
        : "";
    const media = renderPrivacyPreviewMedia(event, evidenceStatus);
    container.innerHTML = `
      <div class="detail-heading">
        <div>
          <p class="eyebrow">告警事件</p>
          <h3>${escapeHtml(event.event_id || "事件详情")}</h3>
        </div>
        <div class="detail-actions">
          ${badge(displayStatus)}
          ${handleButton}
        </div>
      </div>
      <div class="decision-summary" aria-label="事件判断摘要">
        <div class="decision-summary-item">
          <span>风险等级</span>
          <strong>${escapeHtml(currentDecision)}</strong>
        </div>
        <div class="decision-summary-item">
          <span>处置状态</span>
          <strong>${escapeHtml(alertStatusLabel(event.alert_status))}</strong>
        </div>
        <div class="decision-summary-item">
          <span>隐私预览</span>
          <strong>${escapeHtml(evidenceStatus)}</strong>
        </div>
      </div>
      <div class="detail-media">${media}</div>
      ${renderEvidenceRail({
        display_status: displayStatus,
        alert_status: event.alert_status,
        yolo_score: event.yolo_score || candidate.score,
        vlm_label: event.vlm_label,
      })}
      <h4 class="detail-subhead">关键判断</h4>
      <div class="detail-grid evidence-grid">
        ${detailItem("摄像头 ID", event.camera_id)}
        ${detailItem("告警时间", event.created_at)}
        ${detailItem("风险等级", event.risk_label || event.status_label)}
        ${detailItem("处置状态", alertStatusLabel(event.alert_status))}
        ${detailItem("处理账号", event.handled_by)}
        ${detailItem("处理时间", event.handled_at)}
        ${detailItem("YOLO 摘要", event.candidate_summary || candidate.score)}
        ${detailItem("VLM 状态", event.vlm_label || "未分级")}
        ${detailItem("VLM 置信度", vlmConfidence)}
        ${detailItem("VLM 判断理由", verification.reason || verification.failure_reason || "未记录")}
        ${detailItem("可见证据", Array.isArray(verification.visible_evidence) ? verification.visible_evidence.join("；") : "")}
      </div>
      <h4 class="detail-subhead">安全留存</h4>
      <div class="detail-grid evidence-grid">
        ${detailItem("决策来源", event.decision_source)}
        ${detailItem("系统降级", event.system_degraded ? "是" : "否")}
        ${detailItem("上次提醒", event.last_notified_at)}
        ${detailItem("下次提醒", event.next_remind_at)}
        ${detailItem("提醒次数", event.reminder_count)}
        ${detailItem("隐私预览状态", evidenceStatus)}
        ${detailItem("片段时长", event.duration_seconds ? `${event.duration_seconds} 秒` : "未记录")}
      </div>
    `;
  } catch (error) {
    container.innerHTML = `<p class="error">详情加载失败：${escapeHtml(error.message)}</p>`;
  }
}

function renderPrivacyPreviewMedia(event, evidenceStatus) {
  if (event.privacy_preview_status === "ready" && event.privacy_preview_url) {
    return `
      <video
        class="video-player privacy-preview-player"
        controls
        preload="metadata"
        src="${escapeAttr(event.privacy_preview_url)}"
        aria-label="隐私剪影预览视频"
      ></video>
    `;
  }
  if (event.privacy_preview_status === "processing" || event.privacy_preview_status === "pending") {
    return `<div class="camera-placeholder empty-video">隐私视频生成中</div>`;
  }
  if (event.privacy_preview_status === "failed") {
    return `<div class="camera-placeholder empty-video">隐私视频暂不可用，请立即到场核验</div>`;
  }
  return `<div class="camera-placeholder empty-video">${escapeHtml(evidenceStatus || "隐私视频尚未生成")}</div>`;
}

function renderEvidenceRail(event) {
  const status = event.display_status || event.status || "normal";
  const vlmResult = event.vlm_label || "未分级";
  const alertStatus = event.alert_status || "";
  const archived =
    alertStatus === "handled" || status === "handled"
      ? "已处理"
      : status === "no_alarm"
        ? "无告警"
        : status === "pending_detection"
          ? "检测中"
          : "待处置";
  return `
    <div class="event-rail" aria-label="事件证据导轨">
      <div class="rail-step">
        <span>采集</span>
        <strong>已保存</strong>
      </div>
      <div class="rail-step">
        <span>YOLO</span>
        <strong>${formatNumber(event.yolo_score)}</strong>
      </div>
      <div class="rail-step">
        <span>VLM</span>
        <strong>${escapeHtml(vlmResult)}</strong>
      </div>
      <div class="rail-step ${escapeAttr(status)}">
        <span>处置</span>
        <strong>${archived}</strong>
      </div>
    </div>
  `;
}

function renderEvaluationCards(summary) {
  const labelEvaluation = summary.label_evaluation || {};
  const labelBlock = labelEvaluation.available
    ? `
      ${metricCard("Precision", labelEvaluation.precision)}
      ${metricCard("Recall", labelEvaluation.recall)}
      ${metricCard("F1", labelEvaluation.f1)}
      ${metricCard("1000 ms 内准确率", labelEvaluation.start_time_accuracy?.within_1000ms)}
      ${metricCard("2000 ms 内准确率", labelEvaluation.start_time_accuracy?.within_2000ms)}
      ${metricCard("平均起始误差 ms", labelEvaluation.start_time_error_ms?.mean_abs)}
    `
    : `<div class="note">当前未提供人工标注文件，因此 Precision、Recall、F1 和时间准确率尚未计算。</div>`;
  const yolo = summary.yolo || {};
  const vlm = summary.vlm || {};
  return `
    <div class="grid-2">
      ${metricCard("可展示案例", summary.displayed_cases)}
      ${metricCard("高风险告警", summary.high_risk)}
      ${metricCard("低风险告警", summary.low_risk)}
      ${metricCard("检测中事件", summary.pending_detection)}
      ${metricCard("无护工告警", summary.no_alarm)}
      ${metricCard("YOLO 平均分", yolo.average_score)}
      ${metricCard("VLM 平均置信度", vlm.average_confidence)}
      ${metricCard("VLM 已分级", vlm.verified_events)}
      ${metricCard("VLM 高风险", vlm.high_risk)}
      ${metricCard("VLM 无告警", vlm.no_alarm)}
      ${metricCard("VLM 低风险", vlm.low_risk)}
    </div>
    <div class="grid-2" style="margin-top: 12px;">${labelBlock}</div>
  `;
}

function renderArchitecture() {
  document.querySelector("#architecture").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Technical architecture</p>
      <h2>技术方案</h2>
      <p>当前原型使用本地视频模拟摄像头输入，后续可接入 RTSP 或真实摄像头。</p>
    </div>
    <div class="pipeline">
      ${[
        ["摄像头视频流", "当前由本地视频目录模拟输入。"],
        ["YOLO 实时候选检测", "pose/person 模型生成内部候选事件片段。"],
        ["事件前后帧缓存", "EventBuffer 保存触发点前后帧。"],
        ["候选事件片段生成", "ClipBuilder 写入内部事件片段和 JSON 元数据。"],
        ["Video VLM 风险分级", "SQLite 队列支持异步 worker 写入高低风险。"],
        ["告警状态机", "SQLite 告警状态优先作为前端分类依据。"],
      ]
        .map(
          ([title, body]) => `
            <article class="pipeline-step">
              <h3>${title}</h3>
              <p>${body}</p>
            </article>
          `,
        )
        .join("")}
    </div>
    <div class="grid-2" style="margin-top: 18px;">
      <article class="panel">
        <h3>已实现链路</h3>
        <ul class="list-plain">
          <li>本地视频模拟摄像头输入。</li>
          <li>YOLO pose/person 模型生成内部候选事件片段。</li>
          <li>事件前后帧缓存、片段写入和 JSON 元数据保存。</li>
          <li>SQLite 队列、异步 VLM worker、风险等级合并和指标报告。</li>
        </ul>
      </article>
      <article class="panel">
        <h3>后续安全扩展方向</h3>
        <ul class="list-plain">
          <li>RTSP / 真实摄像头接入和持续实时视频画面。</li>
          <li>剪影预览、事件视频加密、人体区域隐私保护和授权查看。</li>
          <li>视频哈希校验、多节点备份、删除/篡改检测。</li>
        </ul>
      </article>
    </div>
  `;
}

function getPageFromHash() {
  const match = window.location.hash.match(/^#page=([a-z-]+)$/);
  const page = match ? match[1] : "intro";
  return pages.includes(page) ? page : "intro";
}

function setActivePage(page) {
  const activePage = pages.includes(page) ? page : "intro";
  document.querySelectorAll(".page-view").forEach((section) => {
    section.classList.toggle("active", section.dataset.page === activePage);
  });
  document.querySelectorAll("[data-page-link]").forEach((link) => {
    link.classList.toggle("active", link.dataset.pageLink === activePage);
    if (link.dataset.pageLink === activePage) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  window.scrollTo({ top: 0, behavior: "auto" });
}

function badge(status) {
  const value = status || "normal";
  return `<span class="status-badge ${escapeAttr(value)}">${escapeHtml(labels[value] || value)}</span>`;
}

function metricCard(title, value) {
  return `
    <article class="metric-card">
      <h3>${escapeHtml(title)}</h3>
      <span class="metric-value">${formatNumber(value)}</span>
    </article>
  `;
}

function detailItem(label, value) {
  return `
    <div class="detail-item">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value ?? "未记录")}</strong>
    </div>
  `;
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "0";
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function setAlertFilter(value) {
  state.activeAlertFilter = value;
  resetEventPage("alerts");
  const events = filteredAlerts();
  state.selectedAlertId = events[0]?.event_id || null;
  renderAlertsCenter();
}

function alertStatusLabel(value) {
  if (value === "pending") return "待处理";
  if (value === "handled") return "已处理";
  if (value === "none") return "无护工告警";
  return value || "未记录";
}

function privacyPreviewStatusLabel(value) {
  if (value === "processing") return "隐私视频生成中";
  if (value === "ready") return "隐私视频已生成";
  if (value === "failed") return "隐私视频暂不可用，请立即到场核验";
  return "隐私视频尚未生成";
}

function setShowcaseFilter(value) {
  state.activeShowcaseFilter = value;
  resetEventPage("showcase");
  const cases = filteredShowcaseCases();
  state.selectedShowcaseId = cases[0]?.event_id || null;
  renderShowcaseEvaluation();
}

function selectAlert(eventId) {
  state.selectedAlertId = eventId;
  renderAlertsCenter();
}

function openAlertFromQueue(eventId) {
  state.activeAlertFilter = "all";
  setEventPageForSelection("alerts", eventId);
  state.selectedAlertId = eventId;
  renderAlertsCenter();
  if (window.location.hash === "#page=alerts") {
    setActivePage("alerts");
    return;
  }
  window.location.hash = "#page=alerts";
}

function selectFallEvent(eventId) {
  state.selectedFallEventId = eventId;
  renderFallEvents();
}

function selectReviewAlert(eventId) {
  state.selectedReviewAlertId = eventId;
  renderReviewAlerts();
}

function selectShowcase(eventId) {
  state.selectedShowcaseId = eventId;
  renderShowcaseEvaluation();
}

async function handleEvent(eventId) {
  if (!eventId) return;
  try {
    await fetchJson(`/api/events/${encodeURIComponent(eventId)}/handle`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-Account": "demo_caregiver",
      },
      body: "{}",
    });
    state.reminders = state.reminders.filter((item) => item.event_id !== eventId);
    await loadDynamicData();
  } catch (error) {
    window.alert(`标记已处理失败：${error.message}`);
  }
}

function limitedEvents(events, limitKey) {
  const limit = state.visibleLimits[limitKey] || EVENT_PAGE_SIZE;
  return events.slice(0, limit);
}

function filteredAlerts() {
  return state.alerts.filter((event) => {
    if (state.activeAlertFilter === "all") return true;
    if (state.activeAlertFilter === "handled") {
      return event.alert_status === "handled";
    }
    return (
      event.risk_level === state.activeAlertFilter &&
      event.alert_status !== "handled"
    );
  });
}

function filteredShowcaseCases() {
  return state.showcaseCases.filter((event) => {
    if (state.activeShowcaseFilter === "all") return true;
    if (state.activeShowcaseFilter === "handled") {
      return event.alert_status === "handled";
    }
    return event.risk_level === state.activeShowcaseFilter;
  });
}

function eventsForPageKey(pageKey) {
  const eventGroups = {
    alerts: filteredAlerts,
    fallEvents: () => state.fallEvents,
    reviewAlerts: () => state.reviewAlerts,
    showcase: filteredShowcaseCases,
  };
  const getter = eventGroups[pageKey];
  return typeof getter === "function" ? getter() : [];
}

function pagedEvents(events, pageKey) {
  const page = normalizedEventPage(pageKey, events.length);
  const start = page * EVENT_PAGE_SIZE;
  return events.slice(start, start + EVENT_PAGE_SIZE);
}

function normalizedEventPage(pageKey, totalCount) {
  const pageCount = Math.max(1, Math.ceil(totalCount / EVENT_PAGE_SIZE));
  const current = state.eventPages[pageKey] || 0;
  const page = Math.min(Math.max(current, 0), pageCount - 1);
  state.eventPages[pageKey] = page;
  return page;
}

function renderEventPagination(pageKey, totalCount) {
  const pageCount = Math.ceil(totalCount / EVENT_PAGE_SIZE);
  if (pageCount <= 1) {
    return "";
  }
  const page = normalizedEventPage(pageKey, totalCount);
  const rerenderName = rerenderNameForLimit(pageKey);
  const pageButtons = Array.from({ length: pageCount }, (_, index) => {
    const label = `第 ${index + 1} 页`;
    return `
      <button class="page-button ${index === page ? "active" : ""}" type="button" onclick="setEventPage('${pageKey}', ${index}, '${rerenderName}')" aria-label="${label}" ${index === page ? 'aria-current="page"' : ""}>
        ${index + 1}
      </button>
    `;
  }).join("");
  return `
    <div class="pagination-row" aria-label="列表分页">
      <button class="page-button" type="button" onclick="setEventPage('${pageKey}', ${page - 1}, '${rerenderName}')" ${page === 0 ? "disabled" : ""}>
        上一页
      </button>
      <div class="page-buttons">${pageButtons}</div>
      <button class="page-button" type="button" onclick="setEventPage('${pageKey}', ${page + 1}, '${rerenderName}')" ${page >= pageCount - 1 ? "disabled" : ""}>
        下一页
      </button>
      <span>第 ${page + 1} / ${pageCount} 页，共 ${totalCount} 条</span>
    </div>
  `;
}

function setEventPage(pageKey, page, rerenderName) {
  const events = eventsForPageKey(pageKey);
  const pageCount = Math.max(1, Math.ceil(events.length / EVENT_PAGE_SIZE));
  state.eventPages[pageKey] = Math.min(Math.max(page, 0), pageCount - 1);
  selectFirstEventOnCurrentPage(pageKey);
  const rerender = window[rerenderName];
  if (typeof rerender === "function") {
    rerender();
  }
}

function selectFirstEventOnCurrentPage(pageKey) {
  const event = pagedEvents(eventsForPageKey(pageKey), pageKey)[0] || null;
  setSelectedEventForPageKey(pageKey, event?.event_id || null);
}

function setEventPageForSelection(pageKey, eventId) {
  const events = eventsForPageKey(pageKey);
  const index = events.findIndex((event) => event.event_id === eventId);
  state.eventPages[pageKey] =
    index >= 0 ? Math.floor(index / EVENT_PAGE_SIZE) : 0;
}

function setSelectedEventForPageKey(pageKey, eventId) {
  const selectionSetters = {
    alerts: () => {
      state.selectedAlertId = eventId;
    },
    fallEvents: () => {
      state.selectedFallEventId = eventId;
    },
    reviewAlerts: () => {
      state.selectedReviewAlertId = eventId;
    },
    showcase: () => {
      state.selectedShowcaseId = eventId;
    },
  };
  const setSelection = selectionSetters[pageKey];
  if (typeof setSelection === "function") {
    setSelection();
  }
}

function renderShowMore(limitKey, totalCount, rerenderName) {
  const visibleCount = Math.min(
    state.visibleLimits[limitKey] || EVENT_PAGE_SIZE,
    totalCount,
  );
  if (visibleCount >= totalCount) {
    return "";
  }
  const remaining = totalCount - visibleCount;
  return `
    <div class="show-more-row">
      <button type="button" onclick="showMore('${limitKey}', '${rerenderName}')">
        查看更多（剩余 ${remaining} 条）
      </button>
      <span>当前显示 ${visibleCount} / ${totalCount}</span>
    </div>
  `;
}

function showMore(limitKey, rerenderName) {
  state.visibleLimits[limitKey] =
    (state.visibleLimits[limitKey] || EVENT_PAGE_SIZE) + EVENT_PAGE_SIZE;
  const rerender = window[rerenderName];
  if (typeof rerender === "function") {
    rerender();
  }
}

function resetVisibleLimit(limitKey) {
  state.visibleLimits[limitKey] = EVENT_PAGE_SIZE;
}

function startReminderPolling() {
  if (state.reminderPollingStarted) return;
  state.reminderPollingStarted = true;
  pollReminders();
  window.setInterval(pollReminders, 2000);
}

async function pollReminders() {
  try {
    const payload = await fetchJson("/api/reminders");
    const reminders = payload.reminders || [];
    const activeReminders = reminders.filter((reminder) => reminder.alert_status === "pending");
    if (!activeReminders.length) return;
    state.reminders = [...activeReminders, ...state.reminders]
      .filter((item, index, all) => {
        return all.findIndex((other) => other.event_id === item.event_id) === index;
      })
      .slice(0, 6);
    playReminderTone(activeReminders[0].risk_level);
    renderMonitoring();
    renderAlertsCenter();
  } catch (error) {
    console.warn("提醒轮询失败", error);
  }
}

function playReminderTone(riskLevel) {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) return;
  try {
    const context = new AudioContext();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = riskLevel === "high_risk" ? 880 : 620;
    gain.gain.value = 0.035;
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.16);
    oscillator.onended = () => context.close();
  } catch (error) {
    console.warn("提醒声音不可用", error);
  }
}

function resetEventPage(pageKey) {
  state.eventPages[pageKey] = 0;
}

function rerenderNameForLimit(limitKey) {
  const rerenders = {
    alerts: "renderAlertsCenter",
    fallEvents: "renderFallEvents",
    reviewAlerts: "renderReviewAlerts",
    showcase: "renderShowcaseEvaluation",
  };
  return rerenders[limitKey] || "renderMonitoring";
}

window.renderMonitoring = renderMonitoring;
window.renderAlertsCenter = renderAlertsCenter;
window.renderFallEvents = renderFallEvents;
window.renderReviewAlerts = renderReviewAlerts;
window.renderShowcaseEvaluation = renderShowcaseEvaluation;
window.showMore = showMore;
window.setAlertFilter = setAlertFilter;
window.setShowcaseFilter = setShowcaseFilter;
window.setEventPage = setEventPage;
window.selectAlert = selectAlert;
window.openAlertFromQueue = openAlertFromQueue;
window.selectFallEvent = selectFallEvent;
window.selectReviewAlert = selectReviewAlert;
window.selectShowcase = selectShowcase;
window.handleEvent = handleEvent;
