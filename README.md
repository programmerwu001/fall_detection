# Fall Edge Gateway

面向养老院等养老照护场景的摔倒事件检测、隐私保护与防删除取证系统。

当前重点已经落到摔倒检测链路，后续会继续补充视频隐私保护、存储防篡改、防删除取证等安全模块。

## 项目定位

普通摔倒检测系统通常只关注“是否检测到老人摔倒”。本项目的目标更偏安全方向：在检测到疑似摔倒事件后，对事件视频进行受控保存、隐私保护和防删除处理，避免被拍摄者隐私被随意查看，也降低养老院方删除视频、销毁证据的风险。

当前项目规划包含三部分：

1. **摔倒检测**
   使用 YOLO 进行高召回的疑似摔倒候选检测，再使用 Video VLM 对候选片段进行复核。

2. **隐私保护**
   后续计划对事件视频进行加密存储，或对人体区域进行打码处理。视频查看需要经过授权，例如本人、家属、警方或其他具备权限的角色。

3. **防篡改与防删除**
   后续计划参考专利 CN116260926A 中的方案，实现事件视频的防篡改、防删除和证据留存能力。

## 当前状态

当前已经实现的是摔倒检测主链路，达到原型演示阶段：

- 从本地视频目录扫描输入视频。
- 将本地视频目录模拟为一路摄像头帧流，并支持视频边界软重置或严格连续拼接。
- 使用 YOLO pose/person 模型检测疑似摔倒候选。
- 缓存事件触发点前后的帧。
- 可选使用 MiniCPM-V 等 Video VLM 复核候选事件。
- 将确认事件保存为视频片段和 JSON 元数据。
- 提供部分单元测试，覆盖参数解析、事件缓存、候选检测评分、VLM 响应解析等逻辑。

尚未完成：

- 尚未完成正式数据集评估和指标统计。
- 尚未接入真实摄像头或 RTSP 视频流。
- 隐私保护模块尚未实现。
- 防篡改、防删除模块尚未实现。
- 依赖文件 `requirements.txt` 尚未整理完善。

## 当前检测流程

```text
本地视频目录
  -> FileVideoSource 读取视频帧
  -> EventBuffer 缓存事件前后帧
  -> YoloCandidateDetector 生成疑似摔倒候选
  -> VideoVLMVerifier 复核候选事件
  -> ClipBuilder 保存事件视频和元数据
  -> data/events 输出结果
```

其中 YOLO 阶段负责快速筛选疑似事件，VLM 阶段负责减少误报，例如区分摔倒、坐下、弯腰、躺床、遮挡不清等情况。

## 设计文档

- [YOLO 和 VLM 异步处理需求文档](docs/requirements/yolo-vlm-async-processing.md)

## 目录结构

```text
fall_edge_gateway/
├── run_gateway.py                  # 摔倒检测流水线入口
├── config.py                       # 全局路径配置
├── configs/
│   └── detection_config.json       # 检测流程配置
├── services/
│   ├── file_video_source.py        # 本地视频帧流读取
│   ├── event_buffer.py             # 事件前后帧缓存
│   ├── yolo_candidate_detector.py  # YOLO 疑似摔倒候选检测
│   ├── video_vlm_verifier.py       # Video VLM 复核
│   └── clip_builder.py             # 事件视频与元数据保存
├── data/
│   ├── test_videos/                # 测试视频数据
│   └── events/                     # 事件输出目录
├── tests/                          # 单元测试
└── yolo26n-pose.pt                 # YOLO 姿态模型权重
```

## 环境准备

当前开发环境使用 Python 和 Conda。已验证过的主要依赖包括：

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

- YOLO 姿态模型：`yolo26n-pose.pt`
- Video VLM：`MiniCPM-V-4.6`

默认配置中，MiniCPM-V 模型路径指向：

```text
E:\viedo_vlm\models\MiniCPM-V-4.6
```

如果该本地目录不存在，代码会回退到 `openbmb/MiniCPM-V-4.6`。

## 配置说明

主配置文件位于：

```text
configs/detection_config.json
```

常用配置项：

| 配置项 | 作用 |
|---|---|
| `video_dir` | 输入视频目录；目录下视频会按文件名顺序作为一路模拟摄像头输入 |
| `output_dir` | 事件输出目录 |
| `max_videos` | 该模拟摄像头最多串联处理的视频数量 |
| `video_boundary_policy` | 视频边界策略；`soft_reset` 默认认为相邻视频无关，只保持摄像头 ID，帧号/时间戳/缓存/冷却/跟踪状态都会重置；`continuous` 严格连续拼接 |
| `fps_limit` | 限制检测帧率，降低计算量 |
| `pre_event_seconds` | 保存事件触发前多少秒 |
| `post_event_seconds` | 保存事件触发后多少秒 |
| `yolo_model` | YOLO 模型权重路径 |
| `yolo_device` | YOLO 推理设备，例如 `"0"` 表示 GPU 0 |
| `candidate_threshold` | YOLO 疑似摔倒候选阈值 |
| `skip_vlm` | 是否跳过 VLM 复核 |
| `vlm_model` | VLM 模型路径或 Hugging Face 模型名 |
| `vlm_backend` | VLM 后端，当前支持 `transformers` 和 `minicpm_chat` |
| `vlm_confidence_threshold` | VLM 确认摔倒的最低置信度 |
| `save_review` | 是否保存需要人工复核的事件 |
| `save_rejected` | 是否保存被拒绝的事件 |

## 运行方式

使用默认配置运行：

```powershell
conda run -n DL1 python run_gateway.py --config configs/detection_config.json
```

只调试 YOLO 候选检测，跳过 VLM 复核：

```powershell
conda run -n DL1 python run_gateway.py --config configs/detection_config.json --skip-vlm --max-videos 1
```

覆盖输入视频目录：

```powershell
conda run -n DL1 python run_gateway.py --video-dir data\test_videos\multiple\dataset\chute24 --max-videos 1
```

如果目录下视频确实是同一摄像头连续切分出的片段，可以启用严格连续拼接：

```powershell
conda run -n DL1 python run_gateway.py --video-boundary-policy continuous
```

提高日志详细程度：

```powershell
conda run -n DL1 python run_gateway.py --log-level DEBUG
```

## 前端演示

项目提供一个本地只读前端，用于展示养老照护摄像头实时摔倒监测项目介绍、实时监测工作台、告警中心、摔倒事件记录、待复核告警、案例回放与评估和技术方案。

启动前端：

```powershell
python app.py
```

默认访问地址：

```text
http://127.0.0.1:8000
```

首页是项目介绍页；实时监测工作台作为同级导航模块提供。当前原型使用本地视频模拟摄像头输入，因此实时监测页摄像头卡片使用占位画面，不展示持续实时流。后续接入 RTSP 或真实摄像头后，可以替换为实时视频画面。

前端读取现有的 `data/events`、`data/records.db`、`configs/detection_config.json` 和指标报告数据。第一版前端不会启动 YOLO 或 VLM 推理任务，也不会修改事件数据。

事件分类规则：

- `confirmed_fall` 显示在“摔倒事件记录”，并让对应摄像头显示最高风险状态。
- `candidates` 和 `need_human_review` 显示在“待复核告警”。
- `rejected` 默认不展示事件列表，只在“案例回放与评估”中统计数量。
- 当 `data/records.db` 存在时，前端优先使用 SQLite 中的事件状态。

## 输出结果

事件输出目录默认为：

```text
data/events/
```

保存结构大致如下：

```text
data/events/
└── file_cam_001/
    └── 20260616/
        ├── event_1.mp4
        ├── event_1.json
        ├── event_2.mp4
        └── event_2.json
```

同一摄像头同一天内按已保存 mp4 数量递增命名；事件类别、候选信息和 VLM 复核结果只记录在 JSON 元数据中。

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
- 跳过 VLM 时的调试保存逻辑

## 当前局限

当前摔倒检测部分还不能视为最终完成版本，主要原因是缺少系统性实验验证：

- 还没有形成公开数据集上的 Precision、Recall、F1 等指标。
- 还没有整理误报、漏报案例。
- 还没有完成 YOLO-only 和 YOLO+VLM 的对比实验。
- 还没有针对不同摄像头角度、多人遮挡、躺床、坐下、弯腰等场景做稳定性评估。
- VLM 推理速度和显存占用还需要进一步验证。

因此当前更适合作为“检测链路已跑通”的工程原型，而不是最终实验结论。

## 后续计划

### 1. 摔倒检测补充验证

- 整理 5 到 10 个代表性视频样例，形成初步效果表。
- 在公开跌倒数据集上跑通批量评估。
- 统计 Precision、Recall、F1、误报率和漏报率。
- 调整 YOLO 候选阈值、事件冷却时间、VLM 采样帧数和置信度阈值。
- 形成用于汇报和论文写作的实验记录。

### 2. 隐私保护模块

待确定具体方案，候选方向包括：

- 事件视频加密存储。
- 人体区域打码。
- 分角色授权查看。
- 授权解密和访问日志记录。

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

> 
