import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticFrontendTest(unittest.TestCase):
    def test_sections_are_page_views_with_page_ids(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

        for page_id in (
            "intro",
            "monitoring",
            "alerts",
            "fall-events",
            "review-alerts",
            "showcase-evaluation",
            "architecture",
        ):
            self.assertIn(f'id="{page_id}" class="page-view', html)
            self.assertIn(f'data-page="{page_id}"', html)

    def test_navigation_uses_page_hashes_not_section_anchors(self):
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

        self.assertIn('href="#page=intro"', html)
        self.assertIn('href="#page=monitoring"', html)
        self.assertNotIn('href="#monitoring"', html)

    def test_javascript_controls_active_page_visibility(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("const pages =", js)
        self.assertIn("function setActivePage", js)
        self.assertIn("window.addEventListener(\"hashchange\"", js)

    def test_event_lists_have_pagination_controls(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("const EVENT_PAGE_SIZE = 4", js)
        self.assertIn("eventPages", js)
        self.assertIn("function pagedEvents", js)
        self.assertIn("function renderEventPagination", js)
        self.assertIn("function setEventPage", js)

    def test_event_table_uses_pagination_not_show_more(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        table_start = js.index("function renderEventTable")
        table_end = js.index("async function renderEventDetail")
        event_table = js[table_start:table_end]

        self.assertIn("pagedEvents", event_table)
        self.assertIn("renderEventPagination", event_table)
        self.assertIn("paginated-list", event_table)
        self.assertNotIn("renderShowMore", event_table)

    def test_css_supports_paginated_event_lists(self):
        css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

        self.assertIn(".paginated-list", css)
        self.assertIn(".pagination-row", css)
        self.assertIn(".page-button", css)

    def test_intro_copy_presents_camera_project_not_local_video_mock(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        intro_start = js.index("function renderIntro()")
        intro_end = js.index("function renderMonitoringLoading()")
        intro = js[intro_start:intro_end]

        self.assertIn("摄像头风险监测原型", intro)
        self.assertNotIn("本地视频模拟摄像头输入", intro)
        self.assertNotIn("本地视频", intro)
        self.assertNotIn("模拟摄像头", intro)

    def test_css_hides_inactive_pages(self):
        css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

        self.assertIn(".page-view", css)
        self.assertIn(".page-view.active", css)
        self.assertIn(".show-more-row", css)

    def test_css_uses_care_duty_desk_design_tokens(self):
        css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("--canvas: #f5f7f4", css)
        self.assertIn("--care-green: #2d6a4f", css)
        self.assertIn("--alarm-red: #c4312d", css)
        self.assertIn("--video-black: #111417", css)
        self.assertIn(".surveillance-wall", css)

    def test_javascript_renders_monitor_wall_and_evidence_rail(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("renderEvidenceRail", js)
        self.assertIn("surveillance-wall", js)
        self.assertIn("event-rail", js)
        self.assertIn("事件证据导轨", js)

    def test_detail_panel_uses_evidence_first_language(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        detail_start = js.index("async function renderEventDetail")
        detail_end = js.index("function renderEvaluationCards")
        detail = js[detail_start:detail_end]

        self.assertIn("告警事件", detail)
        self.assertIn("关键判断", detail)
        self.assertIn("安全留存", detail)
        self.assertIn("event.vlm_label", detail)
        self.assertNotIn("verification.result", detail)

    def test_latest_alerts_open_visible_alert_detail(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        latest_start = js.index("function renderLatestAlerts")
        latest_end = js.index("function renderQueueSummary")
        latest_alerts = js[latest_start:latest_end]

        self.assertIn("openAlertFromQueue", latest_alerts)
        self.assertIn("function openAlertFromQueue", js)
        self.assertIn("window.openAlertFromQueue", js)
        self.assertIn('state.activeAlertFilter = "all"', js)

    def test_detail_panel_has_triage_summary(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        detail_start = js.index("async function renderEventDetail")
        detail_end = js.index("function renderEvaluationCards")
        detail = js[detail_start:detail_end]

        self.assertIn("decision-summary", detail)
        self.assertIn("风险等级", detail)
        self.assertIn("处置状态", detail)
        self.assertIn("隐私预览", detail)

    def test_detail_panel_only_renders_controlled_privacy_preview_video(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        detail_start = js.index("async function renderEventDetail")
        detail_end = js.index("function renderEvaluationCards")
        detail = js[detail_start:detail_end]

        self.assertIn("隐私视频尚未生成", detail)
        self.assertIn("隐私视频生成中", detail)
        self.assertIn("隐私视频暂不可用，请立即到场核验", detail)
        self.assertIn("event.privacy_preview_url", detail)
        self.assertIn("<video", detail)
        self.assertIn('src="${escapeAttr(event.privacy_preview_url)}"', detail)
        self.assertNotIn("event.media_url", detail)
        self.assertNotIn("event.clip_path", detail)

    def test_frontend_does_not_display_internal_legacy_status_labels(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertNotIn("confirmed_fall", js)
        self.assertNotIn("need_human_review", js)
        self.assertNotIn("candidates", js)
        self.assertNotIn("确认摔倒", js)
        self.assertNotIn("人工复核", js)
        self.assertNotIn("人工审核", js)

    def test_reminder_polling_filters_non_pending_reminders_before_alerting(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        polling_start = js.index("async function pollReminders")
        polling_end = js.index("function playReminderTone")
        polling = js[polling_start:polling_end]

        self.assertIn('reminder.alert_status === "pending"', polling)
        self.assertIn("activeReminders", polling)
        self.assertIn("playReminderTone(activeReminders[0].risk_level)", polling)

    def test_css_supports_polished_alert_detail_layout(self):
        css = (ROOT / "static" / "styles.css").read_text(encoding="utf-8")

        self.assertIn(".decision-summary", css)
        self.assertIn(".detail-panel.sticky-detail", css)
        self.assertIn(".event-row::before", css)


if __name__ == "__main__":
    unittest.main()
