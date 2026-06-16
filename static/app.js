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
  visibleLimits: {
    latestAlerts: 8,
    alerts: 8,
    fallEvents: 8,
    reviewAlerts: 8,
    showcase: 8,
  },
};

const labels = {
  confirmed_fall: "已确认摔倒",
  need_human_review: "待复核",
  candidates: "疑似摔倒",
  rejected: "已过滤误报",
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

const EVENT_PAGE_SIZE = 8;

const reviewStatuses = new Set(["candidates", "need_human_review"]);

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
  } catch (error) {
    document.querySelectorAll(".dynamic-target").forEach((node) => {
      node.innerHTML = `<p class="error">数据加载失败：${escapeHtml(error.message)}</p>`;
    });
  }
}

async function fetchJson(url) {
  const response = await fetch(url);
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
          把照护现场的摄像头画面组织成值班态势：先发现疑似摔倒，
          再用 Video VLM 复核，并把回放、理由和留存状态放进同一份事件证据。
        </p>
        <div class="hero-status-strip" aria-label="系统能力边界">
          <span>YOLO 高召回</span>
          <span>VLM 慢路径复核</span>
          <span>只读事件回放</span>
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
            <strong>待复核</strong>
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
            display_status: "confirmed_fall",
            yolo_score: 0.84,
            vlm_result: "fall_detected",
          })}
        </div>
      </aside>
    </div>
    <div class="capability-grid">
      ${[
        ["摄像头风险监测", "以摄像头和区域为主语组织事件，而不是把结果呈现成视频文件列表。"],
        ["疑似摔倒告警生成", "YOLO 高召回检测候选事件，并保留事件前后帧片段。"],
        ["VLM 复核降低误报", "Video VLM 对候选片段给出判断、置信度和理由。"],
        ["摔倒事件记录与回放", "确认事件进入事件档案，可查看来源视频、片段和元数据。"],
        ["待复核告警管理", "不能明确判断的候选告警单独保留，后续接人工审核流程。"],
        ["隐私保护与可信留存方向", "加密、哈希、防删除和授权查看是后续扩展方向，不作为第一版可操作能力。"],
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
          <li>已实现摄像头风险告警、YOLO 候选检测、事件帧缓存和片段保存。</li>
          <li>已实现异步 VLM 复核、SQLite 状态记录、元数据保存和指标报告。</li>
          <li>多摄像头接入、持续监测画面、隐私加密、哈希校验、防删除证据留存和授权查看属于后续方向。</li>
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
        <p>按摄像头聚合当前事件数据，先看高风险区域，再进入回放证据。</p>
      </div>
      <div class="desk-clock">
        <span>值班视图</span>
        <strong>只读演示</strong>
      </div>
    </div>
    <div class="status-strip">
      ${metricCard("摄像头数量", summary.camera_count)}
      ${metricCard("风险摄像头", summary.risk_camera_count)}
      ${metricCard("已确认摔倒", summary.confirmed_fall)}
      ${metricCard("待复核告警", summary.review_alerts)}
      ${metricCard("疑似摔倒", summary.candidate_alerts)}
      ${metricCard("已过滤误报", summary.rejected)}
    </div>
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
                  <span>待复核数量</span>
                  <strong>${formatNumber(camera.pending_review_count)}</strong>
                </div>
                <div class="data-pair">
                  <span>确认事件数量</span>
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
            <button class="alert-item" type="button" onclick="selectAlert('${escapeAttr(event.event_id)}')">
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
        <div class="detail-item"><span>待 VLM 复核</span><strong>${formatNumber(pending)}</strong></div>
        <div class="detail-item"><span>需人工复核</span><strong>${formatNumber(summary.review_alerts)}</strong></div>
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
      <p>正在加载 confirmed、candidates 和 need_human_review 类型告警。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderAlertsCenter() {
  const filters = [
    ["all", "全部告警"],
    ["confirmed_fall", "已确认摔倒"],
    ["need_human_review", "待复核"],
    ["candidates", "疑似摔倒"],
  ];
  const events = state.alerts.filter((event) => {
    return (
      state.activeAlertFilter === "all" ||
      event.display_status === state.activeAlertFilter
    );
  });
  document.querySelector("#alerts").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Alert center</p>
      <h2>告警中心</h2>
      <p>展示系统发现的风险告警，默认不展示已过滤误报。</p>
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
      <div class="detail-panel" id="alert-detail"></div>
    </div>
  `;
  renderEventDetail("#alert-detail", state.selectedAlertId);
}

function renderFallEventsLoading() {
  document.querySelector("#fall-events").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Confirmed events</p>
      <h2>摔倒事件记录</h2>
      <p>正在加载已确认摔倒事件。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderFallEvents() {
  document.querySelector("#fall-events").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Confirmed events</p>
      <h2>摔倒事件记录</h2>
      <p>只展示已确认摔倒事件，形成可回放的事件档案。</p>
    </div>
    <div class="event-layout">
      <div>${renderEventTable(state.fallEvents, state.selectedFallEventId, "selectFallEvent", "fallEvents")}</div>
      <div class="detail-panel" id="fall-event-detail"></div>
    </div>
    <div class="note">该模块后续适合加入访问控制。第一版只展示状态，不实现登录、授权或解密。</div>
  `;
  renderEventDetail("#fall-event-detail", state.selectedFallEventId);
}

function renderReviewAlertsLoading() {
  document.querySelector("#review-alerts").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Review queue</p>
      <h2>待复核告警</h2>
      <p>正在加载候选告警和需人工复核告警。</p>
    </div>
    <div class="dynamic-target"><p class="loading">加载中...</p></div>
  `;
}

function renderReviewAlerts() {
  document.querySelector("#review-alerts").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Review queue</p>
      <h2>待复核告警</h2>
      <p>展示还不能直接归档为摔倒事件、需要 VLM 或人工判断的告警。</p>
    </div>
    <div class="event-layout">
      <div>${renderEventTable(state.reviewAlerts, state.selectedReviewAlertId, "selectReviewAlert", "reviewAlerts")}</div>
      <div class="detail-panel" id="review-alert-detail"></div>
    </div>
    <div class="note">第一版不提供通过、拒绝或提交审核结果按钮，只为后续人工审核流程预留位置。</div>
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
    ["confirmed_fall", "确认摔倒"],
    ["review", "待 VLM/人工复核"],
    ["need_human_review", "需人工复核"],
  ];
  const cases = state.showcaseCases.filter((event) => {
    if (state.activeShowcaseFilter === "all") return true;
    if (state.activeShowcaseFilter === "review") {
      return reviewStatuses.has(event.display_status);
    }
    return event.display_status === state.activeShowcaseFilter;
  });
  document.querySelector("#showcase-evaluation").innerHTML = `
    <div class="section-header">
      <p class="eyebrow">Showcase and evaluation</p>
      <h2>案例回放与评估</h2>
      <p>可点击案例不包含已过滤误报；误报数量只进入指标统计。</p>
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
  const visible = limitedEvents(events, limitKey);
  return `
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
    ${renderShowMore(limitKey, events.length, rerenderNameForLimit(limitKey))}
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
    const media = event.media_url
      ? `<video class="video-player" controls preload="metadata" src="${escapeAttr(event.media_url)}"></video>`
      : `<div class="camera-placeholder empty-video">暂无可播放片段</div>`;
    container.innerHTML = `
      <div class="detail-heading">
        <div>
          <p class="eyebrow">事件证据</p>
          <h3>${escapeHtml(event.event_id || "事件详情")}</h3>
        </div>
        ${badge(event.display_status || event.status)}
      </div>
      <div class="detail-media">${media}</div>
      ${renderEvidenceRail({
        display_status: event.display_status || event.status,
        yolo_score: event.yolo_score || candidate.score,
        vlm_result: verification.result || event.vlm_result,
      })}
      <h4 class="detail-subhead">关键判断</h4>
      <div class="detail-grid evidence-grid">
        ${detailItem("摄像头 ID", event.camera_id)}
        ${detailItem("告警时间", event.created_at)}
        ${detailItem("当前状态", event.status_label)}
        ${detailItem("YOLO candidate 摘要", event.candidate_summary || candidate.score)}
        ${detailItem("VLM 结果", verification.result || event.vlm_result || "未复核")}
        ${detailItem("VLM 置信度", verification.confidence ?? event.vlm_confidence)}
        ${detailItem("VLM 判断理由", verification.reason || verification.failure_reason || "未记录")}
        ${detailItem("可见证据", Array.isArray(verification.visible_evidence) ? verification.visible_evidence.join("；") : "")}
      </div>
      <h4 class="detail-subhead">安全留存</h4>
      <div class="detail-grid evidence-grid">
        ${detailItem("来源模拟视频", event.source_uri)}
        ${detailItem("Clip 路径", event.clip_path)}
        ${detailItem("隐私状态", status.privacy_status || event.privacy_label)}
        ${detailItem("完整性状态", status.integrity_status || event.integrity_label)}
        ${detailItem("留存状态", status.retention_status || event.retention_label)}
        ${detailItem("片段时长", event.duration_seconds ? `${event.duration_seconds} 秒` : "未记录")}
      </div>
    `;
  } catch (error) {
    container.innerHTML = `<p class="error">详情加载失败：${escapeHtml(error.message)}</p>`;
  }
}

function renderEvidenceRail(event) {
  const status = event.display_status || event.status || "normal";
  const vlmResult = event.vlm_result || event.verification_result || "未复核";
  const archived = status === "confirmed_fall" ? "已归档" : status === "rejected" ? "已过滤" : "待处置";
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
        <span>归档</span>
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
      ${metricCard("确认摔倒", summary.confirmed_fall)}
      ${metricCard("待复核告警", summary.review_alerts)}
      ${metricCard("疑似摔倒告警", summary.candidate_alerts)}
      ${metricCard("已过滤误报", summary.rejected)}
      ${metricCard("YOLO 平均分", yolo.average_score)}
      ${metricCard("VLM 平均置信度", vlm.average_confidence)}
      ${metricCard("VLM 已复核", vlm.verified_events)}
      ${metricCard("VLM 确认", vlm.confirmed_fall)}
      ${metricCard("VLM 拒绝", vlm.rejected)}
      ${metricCard("VLM 人工复核", vlm.need_human_review)}
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
        ["YOLO 实时候选检测", "pose/person 模型生成疑似摔倒候选。"],
        ["事件前后帧缓存", "EventBuffer 保存触发点前后帧。"],
        ["疑似摔倒告警生成", "ClipBuilder 写入事件片段和 JSON 元数据。"],
        ["Video VLM 复核", "SQLite 队列支持异步 worker 复核。"],
        ["事件记录 / 误报过滤", "SQLite 状态优先作为前端分类依据。"],
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
          <li>YOLO pose/person 模型生成疑似摔倒候选。</li>
          <li>事件前后帧缓存、片段写入和 JSON 元数据保存。</li>
          <li>SQLite 队列、异步 VLM worker、事件状态合并和指标报告。</li>
        </ul>
      </article>
      <article class="panel">
        <h3>后续安全扩展方向</h3>
        <ul class="list-plain">
          <li>RTSP / 真实摄像头接入和持续实时视频画面。</li>
          <li>事件视频加密、人体区域隐私保护和授权查看。</li>
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
  resetVisibleLimit("alerts");
  const events = state.alerts.filter((event) => {
    return value === "all" || event.display_status === value;
  });
  state.selectedAlertId = events[0]?.event_id || null;
  renderAlertsCenter();
}

function setShowcaseFilter(value) {
  state.activeShowcaseFilter = value;
  resetVisibleLimit("showcase");
  const cases = state.showcaseCases.filter((event) => {
    if (value === "all") return true;
    if (value === "review") return reviewStatuses.has(event.display_status);
    return event.display_status === value;
  });
  state.selectedShowcaseId = cases[0]?.event_id || null;
  renderShowcaseEvaluation();
}

function selectAlert(eventId) {
  state.selectedAlertId = eventId;
  renderAlertsCenter();
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

function limitedEvents(events, limitKey) {
  const limit = state.visibleLimits[limitKey] || EVENT_PAGE_SIZE;
  return events.slice(0, limit);
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
window.selectAlert = selectAlert;
window.selectFallEvent = selectFallEvent;
window.selectReviewAlert = selectReviewAlert;
window.selectShowcase = selectShowcase;
