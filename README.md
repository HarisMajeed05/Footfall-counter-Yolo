# 🚶 Footfall Counter

**AI-powered people counter using YOLO + ByteTrack** — detects, tracks, and counts people entering/exiting a space in real time, with a persistent ID per person so no one is counted twice.

![Python](https://img.shields.io/badge/python-3.10-blue)
![CPU](https://img.shields.io/badge/inference-CPU--only-lightgrey)

---

## What it does

- Detects every person in a video or live webcam feed using a YOLO/RT-DETR model
- Assigns each person a persistent tracking ID (ByteTrack) so the same person is recognized across frames, even through brief occlusion
- Counts **Entry** and **Exit** separately when someone crosses a virtual line — drawn at any angle, not just horizontal
- Overlays bounding boxes, IDs, the counting line, and running totals directly on the video, so every count can be visually verified against the footage
- Runs entirely on CPU — no GPU required

## Why

Manual and sensor-based footfall counting can't reliably tell entry from exit and often double-counts people lingering near an entrance. This project solves that with real detection + tracking: each person is identified individually and counted exactly once per crossing direction.

## Demo

<!-- Add a screenshot or short GIF of the counter running here, e.g.: -->
<!-- ![demo](docs/demo.gif) -->

---

## Table of contents

- [Project structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Drawing a custom counting line](#drawing-a-custom-counting-line)
- [Configuration](#configuration)
- [Roadmap](#roadmap)

---

## Project structure

```
project_root/
├── Models/              # model weight files (yolo11m.pt, rtdetr-l.pt, etc.)
├── Test/                 # your test videos
├── Output/                # annotated output videos land here by default
├── Line_Config/            # saved line.json files from tools/select_line.py
├── tools/
│   └── select_line.py     # interactive line-drawing tool
├── footfall_counter.py     # main script
├── environment.yml
├── requirements.txt
└── README.md
```

## Setup

Requires [Conda](https://docs.conda.io/en/latest/miniconda.html). No GPU needed — everything runs on CPU.

```bash
conda env create -f environment.yml
conda activate footfall-detection
```

## Usage

**Video file, with a live preview window:**

```bash
python footfall_counter.py --source Test/input.mp4 --model yolo11m --show
```

**Video file, save only (no live window, a bit faster):**

```bash
python footfall_counter.py --source Test/input.mp4 --model yolo11m --output Output/result.mp4
```

**Live webcam:**

```bash
python footfall_counter.py --webcam 0 --model yolo11m
```

**Model names** — just the short name, no path or `.pt` extension needed:

```
--model yolo11x    -> resolves to Models/yolo11x.pt
--model yolo11l    -> resolves to Models/yolo11l.pt
--model yolo11m    -> resolves to Models/yolo11m.pt
--model yolo11s    -> resolves to Models/yolo11s.pt
--model yolo11n    -> resolves to Models/yolo11n.pt
--model rtdetr-l   -> resolves to Models/rtdetr-l.pt
```

If the weight file isn't already in `Models/`, it downloads there automatically on first run.

If `--output` isn't given, it auto-saves to `Output/<source-filename>_output.mp4`.

## Drawing a custom counting line

By default the line is horizontal at 50% frame height (`--line-ratio` moves it up/down). For a line at any angle — matching a real doorway, gate, or corridor at whatever camera angle you have:

**1. Draw it once, on the video's first frame:**

```bash
python tools/select_line.py --source Test/input.mp4
```

| Control | Action |
|---|---|
| Left-click | Place a point (2 points define the line) |
| `s` | Save (defaults to `Line_Config/line.json`) |
| `u` | Undo last point |
| `q` | Cancel without saving |

**2. Use it when running the counter:**

```bash
python footfall_counter.py --source Test/input.mp4 --model yolo11m --show --line-config Line_Config/line.json
```

For a live webcam, no separate step is needed — a window opens automatically before detection starts, prompting you to click 2 points (ENTER/SPACE to confirm, `r` to reset, `q`/ESC to abort).

## Configuration

| Flag | Description | Default |
|---|---|---|
| `--source` | Path to a video file | — |
| `--webcam` | Webcam index instead of `--source` | — |
| `--output` | Output video path | auto: `Output/<name>_output.mp4` |
| `--model` | Model name — `yolo11n` / `yolo11s` / `yolo11m` / `yolo11l` / `rtdetr-l` | `yolo11m` |
| `--conf` | Detection confidence threshold | `0.3` |
| `--line-ratio` | Horizontal line position, `0`=top `1`=bottom | `0.5` |
| `--line-config` | Path to a `line.json` for a custom-angle line (overrides `--line-ratio`) | none |
| `--show` | Show a live preview window while processing | off |
| `--imgsz` | Inference resolution — higher improves recall on small/crowded people, at the cost of speed | `1280` |
| `--iou` | NMS IoU threshold — lower it if overlapping people get merged into one box | `0.5` |
| `--max-det` | Max detections per frame — a safety net for very dense crowds | `1000` |

**Model choice**, roughly fastest → most accurate: `yolo11n` < `yolo11s` < `yolo11m` (default) < `yolo11l` < `rtdetr-l` < `yolo11x`. Larger/more accurate models are significantly slower on CPU — benchmark on your own hardware before committing to one for production use.

## Roadmap

- [ ] Multiple counting lines for separate entry/exit gates
- [ ] CSV export of every crossing event with a timestamp
- [ ] Streamlit dashboard UI as an alternative to the CLI + preview window
- [ ] Benchmark script to compare model/imgsz tradeoffs automatically
