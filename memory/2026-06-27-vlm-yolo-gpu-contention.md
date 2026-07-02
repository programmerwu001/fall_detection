# Debug Report: VLM Worker Blocks YOLO GPU Inference

- Date: 2026-06-27
- Symptom: Starting the VLM worker first, then running `run_gateway.py`, made the YOLO gateway appear stuck after several candidate events.
- Evidence:
  - `run_gateway.py` started at 14:59:44 and queued candidate events successfully.
  - `nvidia-smi` showed GPU 0 at about 5963 MiB / 6144 MiB and 100% utilization.
  - `data/records.db` showed 4 VLM jobs marked `failed` with `VLM decision deadline exceeded`.
  - GPU remained saturated after jobs were marked failed, which indicated the timed-out VLM inference thread was still running.
- Root cause: `run_vlm_worker.py` used `ThreadPoolExecutor` to enforce a deadline. Python can time out waiting for the future, but it cannot stop an already-running native/GPU inference thread. The repository could mark the job failed while the VLM generation continued consuming GPU in the background. Running MiniCPM-V and YOLO on the same 6 GB GPU also causes direct memory/compute contention.
- Fix: On VLM timeout, `_verify_with_deadline()` now waits for the in-flight verifier call to finish before returning and marking the job failed, so the worker does not leave hidden GPU work behind.
- Verification:
  - `python -m unittest tests.test_run_vlm_worker` passed.
  - `python -m py_compile run_vlm_worker.py` passed.
  - `nvidia-smi` later showed GPU back to about 411 MiB / 6144 MiB and 0% utilization.
- Remaining operational constraint: A single 6 GB GPU is not enough to run MiniCPM-V-4.6 and YOLO comfortably at the same time. Use separate devices, run YOLO-only first, or run VLM after the gateway queues candidates.
