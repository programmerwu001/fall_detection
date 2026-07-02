# Fall Edge Gateway

面向养老院等养老照护场景的摔倒事件检测、隐私保护与防删除取证系统。

当前代码已经覆盖本地视频摔倒检测、SQLite 异步 VLM 复核、护工告警与重复提醒、本地前端演示、隐私预览原型和指标报告。事件视频加密存储、角色授权访问、完整性哈希、防删除取证等能力仍属于后续安全模块。

## 项目定位

普通摔倒检测系统通常只关注“是否检测到老人摔倒”。本项目的目标更偏安全方向：在检测到疑似摔倒事件后，对事件视频进行受控保存、隐私保护和防删除处理，避免被拍摄者隐私被随意查看，也降低养老院方删除视频、销毁证据的风险。

当前项目规划包含三部分：

1. **摔倒检测**
   使用 YOLO 进行高召回的疑似摔倒候选检测，再使用 Video VLM 对候选片段进行同步或异步复核。

2. **隐私保护**
   当前已实现面向演示的隐私预览：原始事件片段保存在私有目录，前端不直接暴露原始视频，只在隐私预览生成完成后通过受控 `/media/` URL 播放人体剪影版 `privacy_preview.mp4`。加密存储、分角色授权查看和访问日志仍是后续计划。

3. **防篡改与防删除**
   后续计划参考专利 CN116260926A 中的方案，实现事件视频的防篡改、防删除和证据留存能力。

## 当前状态

当前已经实现的是可演示的摔倒检测和告警主链路：

- 从本地视频目录扫描输入视频。
- 将本地视频目录模拟为一路摄像头帧流，并支持视频边界软重置或严格连续拼接。
- 使用 YOLO pose/person 模型检测疑似摔倒候选。
- 缓存事件触发点前后的帧。
- 默认开启异步 VLM：网关保存 YOLO 候选事件并写入 SQLite 队列，`run_vlm_worker.py` 单独领取任务并写回复核结果。
- 仍支持 `--no-async-vlm` 在网关进程内同步调用 VLM。
- 通过 SQLite 保存事件状态、VLM 任务、风险等级、告警状态、处理状态、重复提醒时间和隐私预览任务。
- 将 `confirmed_fall` 映射为高风险告警，将 `need_human_review` 或 VLM 失败/超时映射为低风险告警，将 `rejected` 映射为无护工告警。
- 原始事件片段写入 `data/private_events`；`save_debug_raw_event_copy=true` 时才在 `data/events` 保留本地调试镜像。
- `run_privacy_preview_worker.py` 可为待告警事件生成剪影隐私预览，输出到 `data/privacy_previews/<event_id>/privacy_preview.mp4`。
- 本地前端展示项目介绍、实时监测工作台、告警中心、高风险告警、低风险告警、案例回放与评估、技术方案，并支持告警重复提醒和“已处理”状态。
- 提供指标报告、CAUCAFall 视频级评估和本地测试数据清理脚本。
- 单元测试覆盖参数解析、事件缓存、候选检测评分、VLM worker、SQLite 队列、告警策略、隐私预览、前端 API/静态页面、指标报告等逻辑。

尚未完成：

- 尚未接入真实摄像头或 RTSP 视频流。
- 隐私保护目前只是剪影预览原型，尚未实现加密存储、分角色授权、访问审计。
- 防篡改、防删除、证据留存清单仍未实现。
- 公开数据集上的系统性 Precision、Recall、F1、误报/漏报分析仍需继续补齐。

## 当前检测流程

```text
本地视频目录
  -> FileVideoSource 读取视频帧
  -> EventBuffer 缓存事件前后帧
  -> YoloCandidateDetector 生成疑似摔倒候选
  -> ClipBuilder 保存私有候选片段和可选调试镜像
  -> EventRepository 写入 data/records.db
  -> run_vlm_worker.py 异步复核候选事件
  -> AlertPolicy 写入高/低风险告警、提醒和处理状态
  -> run_privacy_preview_worker.py 生成隐私预览
  -> app.py / static 前端展示结果
```

其中 YOLO 阶段负责快速筛选疑似事件，VLM 阶段负责减少误报，例如区分摔倒、坐下、弯腰、躺床、遮挡不清等情况。默认配置使用异步 VLM；如需单进程调试，可在运行网关时增加 `--no-async-vlm`。

## 设计文档

- [YOLO 和 VLM 异步处理需求文档](docs/requirements/yolo-vlm-async-processing.md)
- [阶段计划](docs/阶段计划.md)

## 目录结构

```text
fall_edge_gateway/
├── app.py                          # 本地前端和告警 API
├── run_gateway.py                  # YOLO 候选检测和事件入库入口
├── run_vlm_worker.py               # 异步 VLM 复核 worker
├── run_privacy_preview_worker.py   # 隐私预览生成 worker
├── generate_metrics_report.py      # 通用指标报告
├── evaluate_caucafall.py           # CAUCAFall 视频级评估
├── config.py                       # 全局路径配置
├── configs/
│   └── detection_config.json       # 检测流程配置
├── scripts/
│   └── reset_test_data.py          # 清理本地演示数据
├── services/
│   ├── alert_policy.py             # VLM 结果到护工告警的映射
│   ├── event_repository.py         # SQLite 事件、VLM 队列和隐私预览队列
│   ├── file_video_source.py        # 本地视频帧流读取
│   ├── event_buffer.py             # 事件前后帧缓存
│   ├── yolo_candidate_detector.py  # YOLO 疑似摔倒候选检测
│   ├── video_vlm_verifier.py       # Video VLM 复核
│   ├── privacy_preview.py          # 人体剪影隐私预览生成
│   ├── frontend_data.py            # 前端数据聚合和脱敏
│   ├── metrics_report.py           # 指标统计
│   └── clip_builder.py             # 私有事件视频与可选调试镜像保存
├── static/                         # 前端页面、样式和交互脚本
├── data/
│   ├── input_videos/test_videos/   # 默认输入视频目录
│   ├── private_events/             # 原始事件片段主存储
│   ├── events/                     # 可选本地调试镜像
│   ├── privacy_previews/           # 剪影隐私预览
│   └── records.db                  # SQLite 事件和任务状态
├── models/                         # 本地模型目录
├── tests/                          # 单元测试
└── requirements.txt                # Python 运行依赖
```

## 环境准备

当前开发环境使用 Python 和 Conda。依赖已经整理在 `requirements.txt`。建议先按机器 CUDA 版本安装 PyTorch，再安装项目依赖，例如：

```powershell
conda run -n DL1 pip install -r requirements.txt
```

已验证过的主要依赖包括：

- Python
- OpenCV
- NumPy
- Pillow
- PyTorch
- Transformers
- Ultralytics
- Accelerate
- Safetensors
- SentencePiece
- Einops
- Timm

当前默认使用的模型：

- YOLO 姿态模型：`models/yolo26n-pose.pt`
- Video VLM：`models/MiniCPM-V-4.6`
- 隐私预览人体分割模型：`models/yolo11n-seg.pt`；如果本地文件不存在，`run_privacy_preview_worker.py` 会使用 Ultralytics 的 `yolo11n-seg.pt` 名称。

`config.py` 的代码默认值支持 `FALL_GATEWAY_VLM_MODEL` 和 `FALL_GATEWAY_PRIVACY_PREVIEW_MODEL` 环境变量；当前 `configs/detection_config.json` 会显式覆盖 VLM 路径为 `models/MiniCPM-V-4.6`，需要改用 Hugging Face 模型名时可修改配置或使用 `--vlm-model`。

## 配置说明

主配置文件位于：

```text
configs/detection_config.json
```

常用配置项：

| 配置项 | 作用 |
|---|---|
| `video_dir` | 输入视频目录；目录下视频会按文件名顺序作为一路模拟摄像头输入 |
| `output_dir` | 本地调试镜像目录；`save_debug_raw_event_copy=true` 时才写入可查看的调试 mp4/json |
| `queue_db_path` | SQLite 事件、VLM 队列、告警和隐私预览队列数据库路径 |
| `recursive` | 是否递归扫描输入视频目录 |
| `max_videos` | 该模拟摄像头最多串联处理的视频数量 |
| `video_boundary_policy` | 视频边界策略；`soft_reset` 默认认为相邻视频无关，只保持摄像头 ID，帧号/时间戳/缓存/冷却/跟踪状态都会重置；`continuous` 严格连续拼接 |
| `fps_limit` | 限制检测帧率，降低计算量 |
| `realtime` | 是否按视频时间戳实时等待播放 |
| `pre_event_seconds` | 保存事件触发前多少秒 |
| `post_event_seconds` | 保存事件触发后多少秒 |
| `buffer_seconds` | 内存帧缓存时长，应大于事件前后窗口 |
| `cooldown_seconds` | 一次事件结束后的冷却时间 |
| `yolo_model` | YOLO 模型权重路径 |
| `yolo_device` | YOLO 推理设备，例如 `"0"` 表示 GPU 0 |
| `yolo_imgsz` | YOLO 输入图像尺寸 |
| `yolo_conf` | YOLO 人体检测置信度阈值 |
| `candidate_threshold` | YOLO 疑似摔倒候选阈值 |
| `skip_vlm` | 是否跳过 VLM 复核；异步模式下会生成低风险演示告警，不入 VLM 队列 |
| `async_vlm` | 是否把候选事件写入 SQLite 队列，由 `run_vlm_worker.py` 异步复核 |
| `save_debug_raw_event_copy` | 是否在 `data/events` 保留本地调试镜像 |
| `high_risk_repeat_seconds` | 高风险告警重复提醒间隔 |
| `low_risk_repeat_seconds` | 低风险告警重复提醒间隔 |
| `vlm_model` | VLM 模型路径或 Hugging Face 模型名 |
| `vlm_backend` | VLM 后端，当前支持 `transformers` 和 `minicpm_chat` |
| `vlm_max_frames` | 每个候选事件最多抽多少帧给 VLM |
| `vlm_max_new_tokens` | VLM 最多生成 token 数 |
| `vlm_temperature` | VLM 采样温度 |
| `vlm_confidence_threshold` | VLM 确认摔倒的最低置信度 |
| `save_review` | 是否保存需要人工复核的事件 |
| `save_rejected` | 是否保存被拒绝的事件 |

## 运行方式

使用默认配置运行：

```powershell
conda run -n DL1 python run_gateway.py --config configs/detection_config.json
```

默认配置启用 `async_vlm=true`，网关只负责 YOLO 候选检测、事件片段保存和入队。启动 VLM worker 领取并复核候选任务：

```powershell
conda run -n DL1 python run_vlm_worker.py --queue-db-path data\records.db
```

为高/低风险告警生成人体剪影隐私预览：

```powershell
conda run -n DL1 python run_privacy_preview_worker.py --queue-db-path data\records.db
```

如果只想在网关进程内同步调用 VLM：

```powershell
conda run -n DL1 python run_gateway.py --config configs/detection_config.json --no-async-vlm
```

只调试 YOLO 候选检测，跳过 VLM 复核：

```powershell
conda run -n DL1 python run_gateway.py --config configs/detection_config.json --skip-vlm --max-videos 1
```

覆盖输入视频目录：

```powershell
conda run -n DL1 python run_gateway.py --video-dir data\input_videos\test_videos --max-videos 1
```

如果目录下视频确实是同一摄像头连续切分出的片段，可以启用严格连续拼接：

```powershell
conda run -n DL1 python run_gateway.py --video-boundary-policy continuous
```

提高日志详细程度：

```powershell
conda run -n DL1 python run_gateway.py --log-level DEBUG
```

清理本地演示数据前可以先预览将删除的路径：

```powershell
conda run -n DL1 python scripts\reset_test_data.py
```

确认删除 `data/events`、`data/private_events`、`data/debug_events_disabled` 和 SQLite 数据库及 sidecar 文件：

```powershell
conda run -n DL1 python scripts\reset_test_data.py --yes
```

## 前端演示

项目提供一个本地前端，用于展示养老照护摄像头实时摔倒监测项目介绍、实时监测工作台、告警中心、高风险告警、低风险告警、案例回放与评估和技术方案。

启动前端：

```powershell
python app.py
```

默认访问地址：

```text
http://127.0.0.1:8000
```

首页是项目介绍页；实时监测工作台作为同级导航模块提供。当前原型后端仍使用本地视频模拟摄像头输入，因此实时监测页摄像头卡片使用占位画面，不展示持续实时流。后续接入 RTSP 或真实摄像头后，可以替换为实时视频画面。

前端读取现有的 `data/events` 调试元数据、`data/records.db`、`configs/detection_config.json` 和指标报告数据。前端不会启动 YOLO 或 VLM 推理任务，但可以通过 `POST /api/events/<event_id>/handle` 将待处理告警标记为“已处理”。提醒轮询通过 `/api/reminders` 领取到期的待处理告警。

前端不会直接提供原始事件视频。只有当隐私预览状态为 `ready` 且预览路径位于 `data/privacy_previews` 下时，页面才会通过 `/media/<token>` 播放 `privacy_preview.mp4`。

事件分类规则：

- `confirmed_fall` 映射为 `high_risk`，显示在“高风险告警”，并让对应摄像头显示高风险状态。
- `need_human_review`、`uncertain`、VLM 失败或超时会映射为 `low_risk`，显示在“低风险告警”。
- `candidates`、`vlm_pending`、`vlm_processing` 显示为检测处理中。
- `rejected` 映射为 `no_alarm`，默认不进入告警列表，只在评估统计中体现。
- 已处理事件保留在列表中，但不再触发重复提醒。
- 当 `data/records.db` 存在时，前端优先使用 SQLite 中的事件状态。

## 输出结果

原始事件片段主存储目录默认为：

```text
data/private_events/
```

保存结构大致如下：

```text
data/private_events/
└── file_cam_001/
    └── 20260616/
        ├── event_1.mp4
        ├── event_1.json
        ├── event_2.mp4
        └── event_2.json
```

同一摄像头同一天内按已保存 mp4 数量递增命名；事件类别、候选信息和 VLM 复核结果只记录在 JSON 元数据和 SQLite 中。

相关输出路径：

| 路径 | 作用 |
|---|---|
| `data/private_events` | 原始事件片段和内部元数据主存储 |
| `data/events` | 可选调试镜像；由 `save_debug_raw_event_copy` 控制 |
| `data/privacy_previews` | 人体剪影隐私预览 |
| `data/debug_events_disabled` | 关闭调试镜像时迁移出的旧调试文件 |
| `data/records.db` | 事件状态、VLM 队列、告警状态、提醒状态和隐私预览队列 |

事件元数据中包含：

- 事件 ID
- 摄像头 ID
- 事件类别
- 视频片段路径
- 帧数量
- 时间范围
- YOLO 候选信息
- VLM 复核结果
- 当前隐私状态
- 当前完整性状态
- 当前留存状态
- SQLite 中还会保存风险等级、告警状态、处理人、处理时间、提醒次数、VLM 状态和隐私预览状态

当前隐私和完整性字段主要用于后续模块衔接，例如：

```json
{
  "privacy_status": "raw_unprotected",
  "integrity_status": "not_hashed",
  "retention_status": "pending_manifest"
}
```

## 指标报告

每次运行检测流程后，可以基于 `data/events` 下保存的事件元数据生成独立指标报告。报告文件用于查看本次运行结果，不需要把具体指标写回 README。

生成默认报告：

```powershell
conda run -n DL1 python generate_metrics_report.py
```

默认输出：

```text
data/events/metrics_summary.json
data/events/metrics_summary.md
```

指定事件目录和输出路径：

```powershell
conda run -n DL1 python generate_metrics_report.py `
  --event-dir data/events `
  --output-json data/events/demo_metrics.json `
  --output-md data/events/demo_metrics.md
```

如果只想统计事件文件，不读取 SQLite 队列状态：

```powershell
conda run -n DL1 python generate_metrics_report.py --no-queue-db
```

报告会统计：

- 事件总数、摄像头数、视频来源数。
- candidates、confirmed_fall、rejected、need_human_review 等类别分布。
- 保存片段的总时长、平均时长、帧数和 FPS。
- YOLO 候选数量、平均候选分数。
- VLM 复核数量、确认/拒绝/人工复核数量、平均置信度。
- SQLite 队列中 pending、processing、done、failed 任务数量。
- 隐私预览队列中 pending、processing、done、failed 任务数量。

Precision、Recall、F1 和时间准确率需要人工标注文件。标注 CSV 至少包含以下列：

```csv
source_uri,event_start_ms,event_end_ms
source_a.mp4,1000,5000
source_b.mp4,2200,7000
```

带标注文件生成评估报告：

```powershell
conda run -n DL1 python generate_metrics_report.py --labels-path data/labels/demo_labels.csv
```

其中时间准确率表示系统候选触发时间落在真实摔倒开始时间附近的比例，目前报告会给出 `within_1000ms` 和 `within_2000ms` 两档结果。未提供标注文件时，报告会明确标记这些准确率类指标未计算。

如果标签是视频级别的 `source_uri,has_fall`，可以使用通用报告的 `--video-labels-path`：

```powershell
conda run -n DL1 python generate_metrics_report.py --video-labels-path data/labels/test_video_labels.csv
```

也可以生成专门的 CAUCAFall 视频级评估报告：

```powershell
conda run -n DL1 python evaluate_caucafall.py --labels-path data\labels\test_video_labels.csv
```

## 测试

运行单元测试：

```powershell
conda run -n DL1 python -m unittest discover -s tests
```

当前测试主要覆盖：

- YOLO 候选评分逻辑
- 候选事件冷却间隔
- VLM JSON 响应解析
- 无法解析响应时回退到人工复核
- 事件缓存窗口
- 配置文件和命令行参数优先级
- 异步 VLM 入队、worker 租约、超时和失败降级
- SQLite 事件状态、告警状态、重复提醒和已处理逻辑
- 隐私预览生成、路径隔离和 worker 生命周期
- 前端数据脱敏、受控媒体 URL、页面导航和告警处理
- 指标报告、视频级评估和测试数据清理

## 当前局限

当前系统还不能视为最终完成版本，主要原因是缺少真实摄像头接入、完整安全模块和系统性实验验证：

- 还没有形成公开数据集上的 Precision、Recall、F1 等指标。
- 还没有整理误报、漏报案例。
- 还没有完成 YOLO-only 和 YOLO+VLM 的对比实验。
- 还没有针对不同摄像头角度、多人遮挡、躺床、坐下、弯腰等场景做稳定性评估。
- VLM 推理速度和显存占用还需要进一步验证。
- 隐私保护目前只是剪影预览；原始片段尚未加密，访问授权和审计日志尚未完成。
- 完整性哈希、防篡改记录、防删除检测和证据恢复流程尚未完成。
- 前端仍是本地演示服务，不是生产鉴权、多用户系统。

因此当前更适合作为“检测、告警和隐私预览链路已跑通”的工程原型，而不是最终实验结论或生产系统。

## 后续计划

### 1. 摔倒检测补充验证

- 整理 5 到 10 个代表性视频样例，形成初步效果表。
- 在公开跌倒数据集上跑通批量评估。
- 统计 Precision、Recall、F1、误报率和漏报率。
- 调整 YOLO 候选阈值、事件冷却时间、VLM 采样帧数和置信度阈值。
- 形成用于汇报和论文写作的实验记录。

### 2. 隐私保护模块

在现有剪影隐私预览基础上继续补齐：

- 事件视频加密存储。
- 分角色授权查看。
- 授权解密和访问日志记录。
- 隐私预览失败时的人工核验和告警流程打磨。

### 3. 防删除与防篡改模块

计划参考专利 CN116260926A 中的方案，逐步实现：

- 事件视频哈希。
- 多节点或冗余存储。
- 删除检测。
- 防篡改记录。
- 证据恢复或一致性校验。

### 4. 系统集成

- 将检测模块、隐私模块、防删除模块串联为完整事件处理流程。
- 补充数据库记录和运行日志。
- 增加真实摄像头输入能力。
- 完善部署说明和演示材料。
