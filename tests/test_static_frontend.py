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

    def test_event_lists_have_display_limits_and_show_more_controls(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn("const EVENT_PAGE_SIZE = 8", js)
        self.assertIn("visibleLimits", js)
        self.assertIn("function limitedEvents", js)
        self.assertIn("function renderShowMore", js)
        self.assertIn("function showMore", js)

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

        self.assertIn("事件证据", detail)
        self.assertIn("关键判断", detail)
        self.assertIn("安全留存", detail)


if __name__ == "__main__":
    unittest.main()
