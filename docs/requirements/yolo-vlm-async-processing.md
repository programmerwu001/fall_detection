# YOLO and VLM Async Processing Requirements

版本：v1.0  
日期：2026-06-05  
适用项目：Fall Edge Gateway

## 1. 背景

当前摔倒检测链路已经跑通：

```text
本地视频源
  -> EventBuffer 缓存事件前后帧
  -> YoloCandidateDetector 生成疑似摔倒候选
  -> VideoVLMVerifier 同步复核候选事件
  -> ClipBuilder 保存事件视频和 JSON 元数据
```

这个流程适合原型验证，但不适合真实部署。主要问题是 VLM 推理耗时高、显存占用大，并且当前 `run_gateway.py` 在 `finalize_event()` 中同步调用 VLM。一旦 VLM 卡住、模型加载失败、GPU OOM 或单个片段推理过慢，摄像头主循环就会被阻塞，导致后续帧处理延迟甚至漏检。

真实部署中，YOLO 应作为实时高召回路径持续运行；VLM 应作为异步慢路径只处理候选片段。两者需要通过持久化事件队列解耦。

## 2. 目标

本需求定义将当前同步链路改造成 YOLO 与 VLM 异步处理架构的执行流程、功能需求、状态机、数据模型、配置项、异常恢复和验收标准。

目标：

- YOLO 检测主链路不等待 VLM 推理结果。
- 疑似摔倒候选先保存为候选事件，再异步进入 VLM 复核队列。
- VLM worker 独立运行，支持失败重试、超时恢复和多 worker 扩展。
- 事件状态可追踪，可恢复，可被后续隐私保护、防篡改、防删除模块接入。
- 边缘设备断电、进程重启或 VLM 崩溃后，未完成任务不会丢失。
- 保留当前本地视频测试流程，同时为 RTSP 多摄像头部署预留接口。

非目标：

- 本阶段不要求实现完整 Web 管理后台。
- 本阶段不要求实现视频加密、防篡改账本或多节点存储。
- 本阶段不要求训练新的 YOLO 或 VLM 模型。
- 本阶段不要求替换现有 `VideoVLMVerifier` 的模型推理逻辑。

## 3. 部署角色

### 3.1 YOLO Gateway

职责：

- 读取本地视频或 RTSP 摄像头流。
- 按配置限制 FPS。
- 维护每路摄像头的事件前后帧缓存。
- 调用 YOLO 生成疑似摔倒候选。
- 收集候选触发点前后若干秒帧。
- 保存候选事件 clip 和 metadata。
- 将 VLM 复核任务写入持久化队列。

约束：

- 不能同步等待 VLM。
- 不能因 VLM worker 不在线而停止摄像头读取。
- 当队列积压达到阈值时，应进入降级策略。

### 3.2 VLM Worker

职责：

- 启动时加载一次 Video VLM 模型。
- 从持久化队列领取 `vlm_pending` 任务。
- 对候选 clip 抽帧。
- 调用 `VideoVLMVerifier` 复核。
- 写回 VLM 结果和事件状态。
- 失败时记录错误并按重试策略处理。

约束：

- 默认单 GPU 上并发为 1。
- 不能重复处理已经完成的事件。
- 进程崩溃后，租约超时的任务必须可被其他 worker 重新领取。

### 3.3 Event Repository

职责：

- 持久化事件基础信息。
- 持久化 VLM job 状态。
- 提供事务化任务领取接口。
- 维护状态流转。
- 支持按状态、摄像头、时间范围查询事件。

建议第一阶段使用 SQLite。后续多设备或中心化部署时，可替换为 Redis、PostgreSQL、RabbitMQ、NATS 或云队列。

## 4. 总体架构

```text
Camera / Video Source
        |
        v
YOLO Gateway Process
  - FileVideoSource / RTSP source
  - EventBuffer
  - YoloCandidateDetector
  - Candidate clip writer
        |
        v
SQLite Event Repository
  - events
  - vlm_jobs
        |
        v
VLM Worker Process
  - lease pending job
  - sample candidate clip
  - VideoVLMVerifier
  - update result and status
        |
        v
Event Output
  - confirmed_fall
  - rejected
  - need_human_review
  - downstream privacy / integrity / retention
```

关键原则：

- YOLO 是实时路径，必须短、稳定、可降级。
- VLM 是慢路径，必须异步、可重试、可恢复。
- 事件 clip 和 metadata 是异步边界，队列中只传路径和结构化信息，不传大图像数组。

## 5. 事件状态机

事件状态必须集中定义，避免在代码中散落字符串。

```text
yolo_candidate
  -> vlm_pending
  -> vlm_processing
  -> confirmed_fall
  -> privacy_pending
  -> integrity_pending
  -> retention_pending
  -> archived

yolo_candidate
  -> vlm_pending
  -> vlm_processing
  -> rejected

yolo_candidate
  -> vlm_pending
  -> vlm_processing
  -> need_human_review

vlm_processing
  -> vlm_pending         # worker 崩溃或租约过期后重新入队
  -> vlm_failed          # 达到最大重试次数
  -> need_human_review   # VLM 不可用但事件仍需保留
```

状态说明：

| 状态 | 含义 |
|---|---|
| `yolo_candidate` | YOLO 已触发疑似摔倒候选，正在收集后置帧或准备落盘。 |
| `vlm_pending` | 候选 clip 已保存，等待 VLM worker 领取。 |
| `vlm_processing` | 某个 worker 已领取任务并正在推理。 |
| `confirmed_fall` | VLM 确认摔倒，且置信度达到阈值。 |
| `rejected` | VLM 判定不是摔倒。 |
| `need_human_review` | 证据不足、VLM 失败或低置信度，需要人工复核。 |
| `vlm_failed` | VLM 多次处理失败，系统保留错误信息。 |
| `privacy_pending` | 已确认事件等待隐私保护处理。 |
| `integrity_pending` | 事件等待哈希、签名或账本登记。 |
| `retention_pending` | 事件等待留存策略或多节点保存。 |
| `archived` | 后续处理完成，进入归档状态。 |

## 6. 执行流程

### 6.1 系统启动

1. 读取 `configs/detection_config.json` 和命令行参数。
2. 初始化日志系统。
3. 初始化事件数据库。
4. 初始化 YOLO 模型。
5. 初始化每路摄像头源。
6. 初始化每路摄像头的 `EventBuffer`。
7. 启动主检测循环。

启动要求：

- 如果 VLM worker 不存在，YOLO Gateway 仍应启动。
- 如果事件数据库不可写，YOLO Gateway 应失败退出，因为候选事件无法可靠保存。
- 如果 YOLO 模型加载失败，YOLO Gateway 应失败退出并输出明确错误。

### 6.2 YOLO 实时检测流程

1. 从摄像头源读取一帧，生成 frame packet。
2. 写入该摄像头的 `EventBuffer`。
3. 如果当前摄像头处于事件冷却期，跳过候选触发。
4. 调用 `YoloCandidateDetector.detect(packet)`。
5. 如果没有候选，继续处理下一帧。
6. 如果有多个候选，选择最高分候选。
7. 创建 `ActiveEvent`，记录候选触发点和事件结束时间。
8. 在后续帧到达时持续追加到 `ActiveEvent.frames`。
9. 到达 `post_event_seconds` 后结束候选收集。
10. 将候选事件保存为 clip 和 metadata。
11. 写入 `events` 表。
12. 写入 `vlm_jobs` 表，状态为 `pending`。
13. 摄像头进入 cooldown。
14. 主循环继续读取下一帧。

要求：

- 第 10 到 12 步必须尽量快，不能调用 VLM。
- 同一候选事件必须具备稳定 `event_id`。
- 候选 clip 保存失败时，不应入队。
- 入队失败时，候选 metadata 应记录 `queue_status=failed`，并增加错误日志。

### 6.3 候选事件落盘流程

候选事件保存路径建议：

```text
data/events/candidates/{camera_id}/{yyyyMMdd}/{event_id}.mp4
data/events/candidates/{camera_id}/{yyyyMMdd}/{event_id}.json
```

候选 metadata 最少包含：

```json
{
  "event_id": "event_file_cam_001_file_cam_001_10_163_10880",
  "camera_id": "file_cam_001",
  "status": "vlm_pending",
  "source_uri": "input video or rtsp url",
  "clip_path": "data/events/candidates/file_cam_001/20260605/event_xxx.mp4",
  "candidate": {},
  "verification": null,
  "privacy_status": "raw_unprotected",
  "integrity_status": "not_hashed",
  "retention_status": "pending_manifest",
  "created_at": "2026-06-05T12:00:00",
  "updated_at": "2026-06-05T12:00:00"
}
```

注意：

- metadata 中的 `candidate` 使用现有 `YoloCandidateDetector` 输出结构。
- `verification` 在 VLM 完成前为 `null`。
- JSON 示例里的时间和 ID 是格式示例，不代表固定值。

### 6.4 VLM Worker 启动流程

1. 读取同一份配置文件。
2. 初始化事件数据库连接。
3. 加载 `VideoVLMVerifier`。
4. 周期性从队列领取任务。
5. 没有任务时 sleep 一段短时间。
6. 收到停止信号时完成当前任务后退出。

要求：

- VLM 模型只在 worker 启动时加载一次。
- worker 必须有 `worker_id`，例如 `vlm-0`。
- worker 应记录模型 ID、backend、max frames、设备信息。

### 6.5 VLM 任务领取流程

任务领取必须具备租约机制：

1. 查询状态为 `pending` 的任务。
2. 查询 `processing` 但 `locked_until` 已过期的任务。
3. 按 priority、created_at 排序选择一条任务。
4. 在同一事务中更新：
   - `status=processing`
   - `locked_by=<worker_id>`
   - `locked_until=now + timeout`
   - `attempts=attempts + 1`
5. 返回任务给 worker。

要求：

- 多 worker 同时运行时，同一任务只能被一个 worker 领取。
- worker 崩溃后，任务必须在 `locked_until` 过期后可被重新领取。
- 如果 `attempts` 超过 `vlm_max_retries`，任务进入 `failed` 或 `need_human_review`。

### 6.6 VLM 复核流程

1. worker 根据 job 读取 event 记录。
2. 确认 clip_path 存在。
3. 从 clip 抽取 `vlm_max_frames` 帧。
4. 调用 `VideoVLMVerifier.verify(candidate, clip_path=...)` 或等价接口。
5. 解析 VLM 返回：
   - `confirmed_fall`
   - `rejected`
   - `need_human_review`
6. 根据 `vlm_confidence_threshold` 做最终保存策略：
   - result 为 `confirmed_fall` 且 confidence 达标，事件状态为 `confirmed_fall`。
   - result 为 `confirmed_fall` 但 confidence 不达标，事件状态为 `need_human_review`。
   - result 为 `rejected`，事件状态为 `rejected`。
   - result 为 `need_human_review`，事件状态为 `need_human_review`。
7. 将 verification 写回 metadata 和数据库。
8. 将 job 标记为 `done`。

要求：

- VLM 异常不能导致 worker 进程直接退出，除非是不可恢复初始化错误。
- 单个任务失败必须记录 `last_error`。
- 低置信度确认不能直接进入 confirmed。

### 6.7 结果归档流程

候选 clip 可以有两种处理方式：

方案 A：保持候选路径不变，仅用数据库和 metadata 表示最终状态。  
优点是写入少，不需要移动文件；缺点是人工查看目录时不直观。

方案 B：VLM 完成后移动到分类目录。

```text
data/events/confirmed_fall/{camera_id}/{yyyyMMdd}/{event_id}.mp4
data/events/rejected/{camera_id}/{yyyyMMdd}/{event_id}.mp4
data/events/need_human_review/{camera_id}/{yyyyMMdd}/{event_id}.mp4
```

推荐第一阶段采用方案 A，降低实现复杂度。需要人工查看目录时，再增加只读索引或导出命令。若采用方案 B，移动文件和更新数据库必须在同一逻辑事务内处理，失败时要能恢复。

### 6.8 失败与恢复流程

VLM 推理失败：

1. 捕获异常。
2. 记录 `last_error`。
3. 如果 attempts 小于最大重试次数，job 回到 `pending`。
4. 如果 attempts 达到最大重试次数，event 进入 `need_human_review`，job 进入 `failed`。

worker 崩溃：

1. 任务停留在 `processing`。
2. `locked_until` 过期。
3. 其他 worker 重新领取任务。

clip 丢失：

1. event 进入 `need_human_review` 或 `vlm_failed`。
2. job 进入 `failed`。
3. metadata 记录 `clip_missing=true`。

数据库不可写：

- YOLO Gateway 应停止入队并输出高优先级错误。
- VLM Worker 应停止处理新任务，避免无法写回结果。

队列积压：

- 当 pending 数超过 `queue_max_pending`，系统进入降级策略。
- 降级策略按配置选择：
  - 仅保留高分候选。
  - 低分候选只写 metadata，不写 clip。
  - 低分候选直接丢弃并记录计数。

## 7. 功能需求

### FR-001 候选事件异步入队

系统必须在 YOLO 候选事件收集完成后保存候选 clip，并创建 VLM job。该流程不得调用 VLM 模型。

验收：

- `skip_vlm=false` 时，YOLO Gateway 仍能在 VLM worker 未启动的情况下继续处理后续视频帧。
- 候选事件 metadata 中 `verification=null`。
- 数据库中生成对应 `vlm_jobs` 记录。

### FR-002 VLM Worker 独立处理任务

系统必须提供独立入口运行 VLM worker。

验收：

- 可以单独运行 `python run_vlm_worker.py --config configs/detection_config.json --worker-id vlm-0`。
- worker 启动后加载 VLM 模型。
- worker 能领取 pending job 并写回结果。

### FR-003 持久化任务队列

系统必须使用持久化队列保存 VLM 任务，第一阶段使用 SQLite。

验收：

- YOLO Gateway 退出后，未处理任务仍存在。
- VLM Worker 重启后能继续处理之前的任务。
- 多个 worker 并发时不会重复领取同一个 job。

### FR-004 状态机可追踪

系统必须记录事件状态变更。

验收：

- 每个事件至少记录当前 `status`。
- 每次 VLM 完成或失败后更新 `updated_at`。
- metadata 和数据库状态保持一致。

### FR-005 VLM 重试和超时

系统必须支持 VLM 任务失败重试和租约超时恢复。

验收：

- VLM 推理抛异常后，attempts 增加。
- 未超过最大重试次数时任务可重新处理。
- 超过最大重试次数时事件进入 `need_human_review` 或 `vlm_failed`。

### FR-006 结果保存策略

系统必须继续支持以下结果：

- `confirmed_fall`
- `rejected`
- `need_human_review`

验收：

- `confirmed_fall` 必须满足 confidence 阈值。
- `save_review` 控制是否保留人工复核事件的输出。
- `save_rejected` 控制是否保留拒绝事件的输出或索引。

### FR-007 保留同步调试模式

系统应保留调试模式，但默认部署应使用异步模式。

验收：

- `--skip-vlm` 仍可用于快速调试 YOLO。
- 可以配置 `async_vlm=false` 使用旧的同步流程做回归对比。
- README 中明确推荐部署使用异步模式。

## 8. 数据模型需求

### 8.1 events 表

```sql
CREATE TABLE events (
  event_id TEXT PRIMARY KEY,
  camera_id TEXT NOT NULL,
  source_uri TEXT NOT NULL,
  clip_path TEXT NOT NULL,
  metadata_path TEXT NOT NULL,
  status TEXT NOT NULL,
  yolo_score REAL,
  candidate_json TEXT NOT NULL,
  verification_json TEXT,
  privacy_status TEXT NOT NULL,
  integrity_status TEXT NOT NULL,
  retention_status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
```

索引：

```sql
CREATE INDEX idx_events_status_created_at ON events(status, created_at);
CREATE INDEX idx_events_camera_created_at ON events(camera_id, created_at);
```

### 8.2 vlm_jobs 表

```sql
CREATE TABLE vlm_jobs (
  job_id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL DEFAULT 100,
  attempts INTEGER NOT NULL DEFAULT 0,
  locked_by TEXT,
  locked_until TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(event_id)
);
```

索引：

```sql
CREATE INDEX idx_vlm_jobs_status_priority_created_at
ON vlm_jobs(status, priority, created_at);
```

## 9. 配置需求

新增配置项：

| 配置项 | 类型 | 默认值 | 说明 |
|---|---:|---:|---|
| `async_vlm` | bool | `true` | 是否启用 VLM 异步处理。 |
| `queue_db_path` | string | `data/records.db` | SQLite 队列和事件数据库路径。 |
| `vlm_worker_id` | string | `vlm-0` | worker 标识。 |
| `vlm_job_poll_seconds` | float | `1.0` | 没有任务时的轮询间隔。 |
| `vlm_job_timeout_seconds` | int | `300` | 单个 VLM job 租约时长。 |
| `vlm_max_retries` | int | `2` | VLM job 最大重试次数。 |
| `queue_max_pending` | int | `100` | pending 任务积压阈值。 |
| `queue_backpressure_policy` | string | `keep_high_score` | 队列积压时的降级策略。 |
| `candidate_output_dir` | string | `data/events/candidates` | 候选事件输出目录。 |
| `archive_by_result` | bool | `false` | VLM 完成后是否按结果移动文件。 |

保留配置项：

- `fps_limit`
- `pre_event_seconds`
- `post_event_seconds`
- `buffer_seconds`
- `cooldown_seconds`
- `yolo_model`
- `yolo_device`
- `candidate_threshold`
- `vlm_model`
- `vlm_backend`
- `vlm_max_frames`
- `vlm_max_new_tokens`
- `vlm_temperature`
- `vlm_confidence_threshold`
- `save_review`
- `save_rejected`

## 10. 模块改造需求

### 10.1 `run_gateway.py`

改造方向：

- 保留参数解析、视频扫描、YOLO 检测、事件缓存逻辑。
- 将 `finalize_event()` 拆成：
  - `finalize_candidate_event()`
  - `verify_event_sync()`，仅同步调试模式使用。
- 异步模式下不创建 `VideoVLMVerifier`。
- 异步模式下调用 repository 创建事件和 job。

### 10.2 `services/clip_builder.py`

改造方向：

- 支持保存候选事件 category，例如 `candidates`。
- 支持 `verification=None`。
- 支持以传入 `event_id` 写入稳定文件名。

### 10.3 `services/event_repository.py`

改造方向：

- 从空模块补全 SQLite repository。
- 提供 schema 初始化。
- 提供事件创建、事件更新、任务创建、任务领取、任务完成、任务失败接口。

必须提供的接口：

```python
create_candidate_event(...)
enqueue_vlm_job(...)
lease_vlm_job(worker_id, lease_seconds)
complete_vlm_job(job_id, verification, final_status)
fail_vlm_job(job_id, error)
mark_expired_jobs_pending(now)
get_queue_stats()
```

### 10.4 `run_vlm_worker.py`

新增入口：

- 加载配置。
- 初始化 repository。
- 初始化 `VideoVLMVerifier`。
- 循环领取任务。
- 调用 VLM。
- 写回结果。
- 支持 graceful shutdown。

### 10.5 `services/video_vlm_verifier.py`

改造方向：

- 优先复用现有 `verify(candidate, frames=None, clip_path=...)` 能力。
- 明确支持从 clip path 抽帧。
- 保持 JSON 解析和错误回退逻辑。

## 11. 运行方式

### 11.1 YOLO Gateway

```powershell
conda run -n DL1 python run_gateway.py --config configs/detection_config.json --async-vlm
```

预期行为：

- 持续读取摄像头或视频帧。
- 发现候选事件后保存到 `data/events/candidates`。
- 将任务写入 `data/records.db`。
- 不等待 VLM 结果。

### 11.2 VLM Worker

```powershell
conda run -n DL1 python run_vlm_worker.py --config configs/detection_config.json --worker-id vlm-0
```

预期行为：

- 启动时加载 MiniCPM-V。
- 领取 pending job。
- 复核候选 clip。
- 更新事件状态。

### 11.3 调试模式

```powershell
conda run -n DL1 python run_gateway.py --config configs/detection_config.json --skip-vlm --max-videos 1
```

预期行为：

- 跳过 VLM。
- YOLO 候选直接按调试策略处理。
- 用于快速验证 YOLO 阈值、事件缓存和 clip 保存。

## 12. 可观测性需求

YOLO Gateway 日志和指标：

- `frames_read_total`
- `yolo_candidates_total`
- `candidate_clips_saved_total`
- `candidate_enqueue_failed_total`
- `queue_pending_count`
- `queue_backpressure_drop_total`
- `camera_fps`

VLM Worker 日志和指标：

- `vlm_jobs_processed_total`
- `vlm_jobs_failed_total`
- `vlm_jobs_retried_total`
- `vlm_job_latency_seconds`
- `vlm_inference_seconds`
- `vlm_confirmed_total`
- `vlm_rejected_total`
- `vlm_review_total`
- `oldest_pending_job_age_seconds`

日志要求：

- 每个事件日志必须包含 `event_id` 和 `camera_id`。
- VLM worker 日志必须包含 `worker_id` 和 `job_id`。
- 异常日志必须包含错误类型和简短原因。

## 13. 性能与资源需求

实时性目标：

- YOLO Gateway 不因单个 VLM job 阻塞。
- 单路摄像头在配置 `fps_limit=15` 时，主循环应稳定处理输入帧。
- 候选事件落盘和入队应在秒级内完成。

GPU 资源：

- 单 GPU 部署时，VLM worker 默认并发为 1。
- YOLO 与 VLM 共用 GPU 时，应通过队列和 worker 并发控制避免同时爆显存。
- 多 GPU 部署时，可将 YOLO 固定到 GPU 0，将 VLM worker 固定到 GPU 1。

存储：

- 候选事件 clip 会增加磁盘占用。
- 系统应提供候选事件清理策略，例如保留最近 N 天或清理 `rejected` 事件。

## 14. 安全与隐私需求

- 事件 clip 默认是未加密原始视频，metadata 必须标记 `privacy_status=raw_unprotected`。
- 在隐私保护模块完成前，不应对外开放未授权访问接口。
- metadata 不应保存账号密码、访问 token 或其他凭据。
- RTSP URL 如包含密码，应在日志中脱敏。
- 后续加密、哈希、账本模块应从事件状态机接入，而不是直接扫描视频目录。

## 15. 测试需求

单元测试：

- repository 初始化 schema。
- 创建候选事件。
- 创建 VLM job。
- lease 同一 job 不会被两个 worker 同时领取。
- lease 过期后 job 可重新领取。
- VLM 成功后写回 verification。
- VLM 失败后 attempts 增加。
- 超过最大重试次数后进入 `need_human_review` 或 `vlm_failed`。
- `async_vlm=true` 时 `finalize_event()` 不调用 VLM。
- `async_vlm=false` 时保留同步验证路径。

集成测试：

- 启动 YOLO Gateway 处理一段测试视频，只生成 pending job，不启动 VLM worker。
- 再启动 VLM worker，确认 pending job 被消费并写回结果。
- 模拟 VLM worker 中途崩溃，确认租约过期后任务可恢复。
- 模拟 clip 缺失，确认事件进入人工复核或失败状态。

回归测试：

```powershell
conda run -n DL1 python -m unittest discover -s tests
```

## 16. 验收标准

本阶段改造完成后，必须满足：

- YOLO Gateway 和 VLM Worker 可以独立启动。
- YOLO Gateway 在 VLM Worker 未启动时仍能持续生成候选事件。
- 候选事件和 VLM job 可在进程重启后恢复。
- VLM Worker 能消费已有候选任务并写回结果。
- 单个 VLM job 失败不会阻塞其他任务。
- 数据库和 metadata 中的事件状态一致。
- `confirmed_fall`、`rejected`、`need_human_review` 三类结果均可被表达。
- 原有单元测试继续通过。
- 新增异步队列和 worker 测试通过。

## 17. 分阶段实施建议

### 阶段 1：持久化事件和队列

- 补全 `event_repository.py`。
- 新增事件和 job schema。
- 新增 repository 单元测试。

### 阶段 2：YOLO Gateway 异步入队

- 拆分 `finalize_event()`。
- 异步模式下保存候选事件并入队。
- 保留同步调试路径。

### 阶段 3：VLM Worker

- 新增 `run_vlm_worker.py`。
- 复用 `VideoVLMVerifier`。
- 写回 verification 和状态。

### 阶段 4：恢复、重试和可观测性

- 增加租约超时恢复。
- 增加 attempts 和失败状态。
- 增加队列统计和日志指标。

### 阶段 5：部署验证

- 用本地测试视频验证完整链路。
- 用多路本地视频模拟多摄像头。
- 接入 RTSP 后验证长时间运行稳定性。

## 18. 设计取舍

选择 SQLite 而不是内存队列：

- SQLite 能在断电或进程重启后保留任务。
- 当前项目是边缘网关原型，SQLite 部署成本最低。
- 缺点是高并发能力有限，但第一阶段 VLM worker 并发很低，足够使用。

选择先保存候选 clip 再入队：

- VLM worker 不需要访问内存帧，进程间边界清晰。
- 崩溃恢复后仍能重新处理候选事件。
- 缺点是候选事件会增加磁盘占用，需要后续清理策略。

选择 VLM worker 独立进程：

- 避免大模型推理阻塞摄像头主循环。
- 方便单独限制 GPU 并发和重启 worker。
- 缺点是需要新增队列、状态和部署脚本。
