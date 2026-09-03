# [Paper Title]

[English] This repository contains the official implementation of **"[Paper Title]"**.

[中文] 本仓库是论文 **《[论文标题]》** 的官方实现代码。

<!-- 建议补充：
- 论文链接 / arXiv / DOI
- 作者姓名与单位
-->

## Overview

This work addresses **[pig/animal] agonistic behavior (fighting) detection under occlusion**, built on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics). We propose three lightweight modules to improve spatial-interaction modeling and occlusion robustness:

| Module | Type | Description |
| --- | --- | --- |
| **DySnakeConv** | Convolution | Dynamic snake convolution, adapts the kernel to the target's morphology. |
| **EUCB-SC** | Upsampling | Channel-shuffle enhanced upsampling block that mixes directional features after upsampling. |
| **CoordAtt** | Attention | Coordinate attention that encodes row/column importance to focus on the interaction region. |

The final model combines all three (`yolo11-dysnake-EUCB-SC-CoordAtt.yaml`).

## Repository Structure

```
├── train.py                      # Training entry point
├── val.py                        # Validation + metrics (params/FLOPs/FPS/mAP)
├── occlusion_final_evaluation.py # Occlusion-robustness evaluation (occluded vs. unoccluded)
├── ultralytics/
│   ├── cfg/models/11puls/        # Paper's model configs (ablation variants)
│   ├── cfg/models/11/yolo11.yaml # Baseline YOLO11 config
│   └── nn/extra_modules/         # Custom modules (DySnakeConv / EUCB-SC / CoordAtt)
└── ...
```

## Installation

Requirements: **Python >= 3.8**, **PyTorch >= 1.8**.

```bash
pip install -r requirements.txt   # or install per pyproject.toml
```

> The custom modules depend only on PyTorch; no extra CUDA kernels are required for the proposed model.

## Dataset

<!-- 请补充数据集的获取方式 / 链接 / 目录结构说明 -->

The dataset should be organized in the YOLO format and referenced by a `data.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: Aggressive
  1: Stand
  2: Lie
  3: Ear biting
  4: Feeding
  # ... (7 classes in total)
```

## Usage

### Train

Edit `data` in `train.py`, then:

```bash
python train.py
```

### Validate (metrics reported in the paper)

Edit `model_path` and `data` in `val.py`, then:

```bash
python val.py
```

### Occlusion-robustness evaluation

Prepare an `occlusion_annotation.xlsx` (columns: `image, bbox_id, class_id, x_center, y_center, width, height, occlusion`), then edit the paths in `occlusion_final_evaluation.py` and run:

```bash
python occlusion_final_evaluation.py
```

## Results

<!-- 请补充论文中的定量结果表格（mAP50 / mAP50-95 / Params / GFLOPs / FPS 等） -->

| Model | mAP50 | mAP50-95 | Params | GFLOPs |
| --- | --- | --- | --- | --- |
| YOLO11 (baseline) | — | — | — | — |
| **Ours (DySnakeConv+EUCB-SC+CoordAtt)** | — | — | — | — |

## Citation

If this work is useful to your research, please cite:

```bibtex
@article{yourkey2026,
  title   = {[Paper Title]},
  author  = {[Authors]},
  journal = {[Journal / Conference]},
  year    = {2026},
  doi     = {[DOI]}
}
```

## Acknowledgements

This implementation is based on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics). The DySnakeConv module is inspired by [DSConv](https://github.com/YaoleiQi/DSCNet); CoordAtt is based on [Coordinate Attention](https://github.com/houqb/CoordAttention).

## License

This project is released under the [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) license, following the base Ultralytics codebase.
