# Unified saved-model evaluation

`evaluate_saved_models.py` is evaluation-only. It discovers existing dense
checkpoints, physically pruned artifacts, and recovery checkpoints; rebuilds
their exact saved architectures; evaluates Bunny; and writes fresh reports to
`<study_root>/unified_inference/`. It does not train, prune, or overwrite pilot
results.

The default CUDA benchmark caches every frame embedding once, then times only
`HNeRVDecoder` (decoder plus RGB head), batch size 1 and FP32, using CUDA Events
with 100 warmups and 300 measured frames. FPS is `1000 / mean_latency_ms`.
Encoder execution, image loading, metric computation, and host/device copies
are outside the timed interval.

Colab command:

```bash
%cd /content/drive/MyDrive/PCDNeRV_HNeRV_Structured_Pruning_Study/PCD-NeRV-v2
!python evaluate_saved_models.py \
  --study_root /content/drive/MyDrive/PCDNeRV_HNeRV_Structured_Pruning_Study \
  --data_path /content/drive/MyDrive/PCDNeRV_HNeRV_Structured_Pruning_Study/data/bunny \
  --device cuda --warmup 100 --timing_runs 300 --expected_frames 132
```

Use `--manifest path/to/models.json` when explicit model selection is desired.
The manifest is a JSON list (or an object with a `models` list) whose entries
may contain `model_label`, `model_type`, `checkpoint`, `artifact`,
`result_json`, and `tag`. Paths may be relative to `--study_root`.

For a quick CPU diagnostic, use `--device cpu --max_models 1 --max_frames 1`.
CPU mode recomputes quality and compute accounting but intentionally leaves
latency/FPS empty, because those fields are defined as CUDA decoder-only
measurements for the study.
