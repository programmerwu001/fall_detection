# Fall Edge Gateway Demo Frontend Design

## 1. Purpose

This document defines the first frontend for `fall_edge_gateway`.

The frontend is for a project mentor to present the project externally and support project pitching. It is not a production nursing-home console and it is not a developer-only debugging tool.

The frontend should help an external audience quickly understand:

- What real problem the project addresses.
- What technical chain has already been implemented.
- What event outputs the current prototype can produce.
- What concrete cases and metrics support the project story.
- Which capabilities are implemented now and which are future extensions.

## 2. Product Positioning

Recommended frontend name:

```text
养老照护摔倒检测与可信事件留存演示平台
```

Recommended subtitle:

```text
基于 YOLO 候选检测与 Video VLM 复核的养老场景风险事件识别原型
```

The first version should feel like a research-and-engineering demonstration platform. It should not look like a marketing landing page, and it should not expose internal controls that can accidentally start long-running GPU jobs.

## 3. Assumptions

- The existing Python detection pipeline remains the source of truth.
- The frontend is read-only in the first version.
- The frontend reads existing local outputs instead of launching YOLO or VLM tasks.
- The project may have no event data yet; empty states must be clear and non-crashing.
- Privacy protection, hashing, anti-deletion, and authorization are future directions. The frontend may describe them, but must not present them as completed interactive features.
- The implementation should avoid large new dependencies. A standard-library Python HTTP server plus static HTML/CSS/JS is preferred.

## 4. Out Of Scope For Version 1

The first version must not include:

- User login.
- Role-based access control.
- Web upload of videos.
- Web-triggered YOLO or VLM execution.
- Real-time camera or RTSP streaming.
- Real-time alert push.
- Actual video encryption or decryption.
- Actual hash-chain verification.
- Delete recovery.
- Multi-node storage operations.
- Admin management workflows.

These can be used as future project directions, but not as working controls in the first frontend.

## 5. Information Architecture

The navigation should contain six sections:

```text
项目概览
场景痛点
技术方案
运行概览
典型案例
评估结果
```

The recommended presentation order is:

```text
项目概览 -> 场景痛点 -> 技术方案 -> 运行概览 -> 典型案例 -> 评估结果
```

## 6. Page Design

### 6.1 项目概览

Goal: help the mentor explain what the project is in under one minute.

Content:

- Project title and subtitle.
- One-sentence description:

```text
面向养老照护场景，自动发现疑似摔倒事件，并保存关键视频片段与事件元数据，为后续隐私保护和可信取证提供基础。
```

- Three capability cards:
  - 摔倒候选自动发现
  - VLM 复核降低误报
  - 事件片段与元数据留存

- Current implementation status:
  - Implemented: local video input, YOLO candidate detection, event frame buffering, async VLM review, event clip storage, metadata storage, metrics report generation.
  - Planned: privacy protection, hash verification, anti-deletion evidence retention, authorized viewing.

This page should avoid dense tables and low-level configuration.

### 6.2 场景痛点

Goal: explain why the project is valuable.

Show four problem blocks:

- 摔倒发现不及时
  老人摔倒后，如果护理人员无法及时发现，可能错过处理窗口。

- 普通监控被动低效
  传统监控主要靠人工回放，不能主动筛选风险事件。

- 照护视频隐私敏感
  养老场景视频涉及个人隐私，需要后续支持受控查看与隐私保护。

- 事件证据可信度不足
  纠纷场景下，普通视频文件可能存在缺失、删除或难以证明完整性的问题。

This page should focus on the problem, not implementation details.

### 6.3 技术方案

Goal: explain how the system works.

Show the current implemented chain:

```text
视频输入
  -> YOLO 姿态/人体检测
  -> 摔倒候选事件
  -> 事件前后帧缓存
  -> 候选视频片段保存
  -> Video VLM 复核
  -> 事件元数据与指标报告
```

Split the page into two areas:

Implemented chain:

- Local video directory input.
- YOLO pose/person model produces fall candidates.
- EventBuffer stores frames before and after the trigger point.
- ClipBuilder writes event clips and JSON metadata.
- SQLite queue supports async VLM worker review.
- Metrics report summarizes outputs and evaluation.

Future security extension directions:

- Event video encryption.
- Human-region privacy protection.
- Video hash verification.
- Multi-node backup.
- Deletion/tamper detection.
- Authorized viewing.

The visual copy must clearly distinguish "已实现" from "后续扩展方向".

### 6.4 运行概览

Goal: answer "what has the current system produced?"

This page should show current runtime/output state and a few key numbers. It must not become a full evaluation table.

Cards and modules:

- 总事件数
- 确认摔倒数
- 需人工复核数
- 候选事件数
- 被拒绝事件数
- 摄像头数量
- 视频来源数量
- VLM queue status: `pending`, `processing`, `done`, `failed`
- Recent events, up to five rows
- Recommended showcase events, prioritizing `confirmed_fall`, then `need_human_review`

Data sources:

```text
data/events/**/*.json
data/records.db
data/events/metrics_summary.json
```

Empty state:

```text
当前尚未生成检测输出，请先运行检测流程。
```

Boundary:

- Do not show Precision, Recall, F1 here.
- Do not show a full event detail here.
- Do not repeat the full metrics report here.

### 6.5 典型案例

Goal: show concrete recognition examples.

This is the most important live-demo page.

List view fields:

- 事件类别
- 摄像头 ID
- 来源视频名
- YOLO 分数
- VLM 结果
- VLM 置信度
- 片段时长
- 创建时间

Filters:

```text
全部
确认摔倒
需人工复核
候选事件
已拒绝
```

Default sorting priority:

```text
confirmed_fall
need_human_review
candidates
rejected
```

Detail view fields:

- Video player for the saved clip.
- Event ID.
- Camera ID.
- Source video.
- Clip path.
- YOLO candidate summary.
- VLM result.
- VLM confidence.
- VLM reason.
- Visible evidence.
- Privacy status.
- Integrity status.
- Retention status.

Status wording:

- `raw_unprotected`: 原始视频，尚未加密
- `not_hashed`: 尚未生成完整性哈希
- `pending_manifest`: 尚未生成留存清单

Boundary:

- Do not show global aggregate metrics here.
- This page is about one event at a time.

### 6.6 评估结果

Goal: provide quantitative support when the audience asks how the prototype performs.

Show:

- Event category distribution.
- Total saved clip duration.
- Average clip duration.
- Total saved frames.
- Average FPS.
- YOLO candidate count.
- Average YOLO score.
- Max YOLO score.
- VLM reviewed event count.
- VLM confirmed, rejected, and human-review counts.
- Average VLM confidence.

If a label CSV exists and has been used by the metrics report, show:

- Precision.
- Recall.
- F1.
- Start-time accuracy within 1000 ms.
- Start-time accuracy within 2000 ms.
- Mean absolute start-time error.

If no label data is available, show:

```text
当前未提供人工标注文件，因此 Precision、Recall、F1 和时间准确率尚未计算。
```

Boundary:

- This page may be more technical than other pages.
- It should be honest about missing labels and current prototype limitations.

## 7. Data Contract

The frontend should be backed by local API endpoints served by `app.py`.

Recommended endpoints:

```text
GET /api/overview
GET /api/events
GET /api/events/<event_id>
GET /api/metrics
GET /api/config
GET /media/<path>
```

### 7.1 `/api/overview`

Returns data for the run overview page.

Suggested response shape:

```json
{
  "has_data": true,
  "counts": {
    "total_events": 0,
    "confirmed_fall": 0,
    "need_human_review": 0,
    "candidates": 0,
    "rejected": 0,
    "cameras": 0,
    "sources": 0
  },
  "queue": {
    "available": false,
    "jobs": {
      "pending": 0,
      "processing": 0,
      "done": 0,
      "failed": 0
    },
    "reason": "queue db does not exist"
  },
  "recent_events": [],
  "showcase_events": []
}
```

### 7.2 `/api/events`

Returns normalized event rows for list display.

Each row should include:

```json
{
  "event_id": "event_xxx",
  "category": "confirmed_fall",
  "category_label": "确认摔倒",
  "camera_id": "file_cam_001",
  "source_name": "video.mp4",
  "source_uri": "E:\\viedo_vlm\\...",
  "clip_path": "E:\\viedo_vlm\\...",
  "media_url": "/media/...",
  "yolo_score": 0.82,
  "vlm_result": "confirmed_fall",
  "vlm_confidence": 0.91,
  "duration_seconds": 6.0,
  "created_at": "2026-06-09T22:00:00"
}
```

### 7.3 `/api/events/<event_id>`

Returns a single normalized event plus detail fields:

```json
{
  "event": {},
  "candidate": {},
  "verification": {},
  "status_explanations": {
    "privacy_status": "原始视频，尚未加密",
    "integrity_status": "尚未生成完整性哈希",
    "retention_status": "尚未生成留存清单"
  }
}
```

### 7.4 `/api/metrics`

Returns the output of `services.metrics_report.build_metrics_report`.

If `data/events/metrics_summary.json` exists and is current enough for the page, it may be used. If it does not exist, the API may build the report in memory without writing files.

### 7.5 `/api/config`

Returns selected public configuration values from `configs/detection_config.json`. It should not expose secrets. Current config values are local paths and thresholds, so it is acceptable to show them for a local demo.

### 7.6 `/media/<path>`

Serves saved event video clips. It must prevent path traversal and only serve files under the project directory or under the configured event output directory.

## 8. Visual Direction

The UI should feel:

- Professional.
- Technical.
- Calm.
- Demonstration-oriented.
- Suitable for a research group presenting to partners.

Recommended layout:

- Top navigation or left sidebar.
- First screen focused on project title and concise value.
- Cards for key capabilities and runtime counts.
- Timeline or pipeline layout for the technical chain.
- Tables for event lists and metrics only where they help scanning.
- Video player visible in the typical-case detail area.

Avoid:

- A pure marketing landing page.
- Heavy gradients or decorative visuals.
- Dense developer-debug tables on the first screen.
- Fake buttons for unimplemented security functions.
- Text claiming privacy or anti-tamper features are completed.

## 9. Success Criteria

The first frontend is successful when:

- `python app.py` starts a local server.
- The browser can open the frontend without a frontend build step.
- All six sections are reachable.
- Empty data directories render useful empty states.
- Existing event JSON files render in the event list.
- Existing event clips can be played from the typical-case page.
- Queue status can be read from `data/records.db` when it exists.
- Metrics can be displayed from saved reports or generated in memory.
- No page starts model inference or modifies event data.
- The UI clearly separates implemented capabilities from future extensions.

## 10. Handoff Notes For Implementers

- Preserve existing detection behavior in `run_gateway.py` and `run_vlm_worker.py`.
- Do not revert existing worktree changes that are unrelated to the frontend.
- Prefer focused helper functions for data loading and normalization.
- Use the existing `services.metrics_report` module instead of duplicating metrics logic.
- Keep frontend dependencies minimal.
- Add unit tests for data normalization and API behavior.
