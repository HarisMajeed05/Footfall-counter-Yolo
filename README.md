# Footfall Counter

Counts people entering and leaving a space from a video file or webcam. It uses YOLO (or RT-DETR) for detection and ByteTrack for tracking, so each person keeps one ID and is counted once per crossing.

![Python](https://img.shields.io/badge/python-3.10-blue)
![CPU](https://img.shields.io/badge/inference-CPU--only-lightgrey)

## Features

- Detects people with a YOLO11 or RT-DETR model
- Tracks each person across frames with ByteTrack
- Counts entries and exits separately when someone crosses a virtual line
- The line can be horizontal or drawn at any angle
- Draws boxes, IDs, the line and running totals on the output video
- Runs on CPU, no GPU needed

## Project structure

```
Models/              model weights (downloaded here on first run)
Test/                test videos
Output/              annotated output videos
Line_Config/         saved line files
tools/select_line.py tool to draw the counting line
footfall_counter.py  main script
environment.yml
requirements.txt
```

## Setup

You need [Conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create -f environment.yml
conda activate footfall-detection
```

## Usage

Video file with a live preview:

```bash
python footfall_counter.py --source Test/input.mp4 --model yolo11m --show
```

Video file, save only:

```bash
python footfall_counter.py --source Test/input.mp4 --model yolo11m --output Output/result.mp4
```

Webcam:

```bash
python footfall_counter.py --webcam 0 --model yolo11m
```

If `--output` is not given, the result is saved to `Output/<source-name>_output.mp4`.

Model names are short names without the `.pt` extension, for example `yolo11m` or `rtdetr-l`. The weights are looked up in `Models/` and downloaded there if missing.

## Drawing a custom counting line

By default the line is horizontal at the middle of the frame. Use `--line-ratio` to move it up or down.

To draw a line at any angle, run the tool on the first frame of your video:

```bash
python tools/select_line.py --source Test/input.mp4
```

| Key | Action |
|---|---|
| Left click | Place a point (two points make the line) |
| `s` | Save to `Line_Config/line.json` |
| `u` | Undo the last point |
| `q` | Quit without saving |

Then pass the saved file to the counter:

```bash
python footfall_counter.py --source Test/input.mp4 --model yolo11m --show --line-config Line_Config/line.json
```

With a webcam, a window opens before detection starts. Click two points, press Enter or Space to confirm, `r` to reset, or `q` / Esc to quit.

## Options

| Flag | Description | Default |
|---|---|---|
| `--source` | Path to a video file | none |
| `--webcam` | Webcam index, used instead of `--source` | none |
| `--output` | Output video path | `Output/<name>_output.mp4` |
| `--model` | `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x` or `rtdetr-l` | `yolo11m` |
| `--conf` | Detection confidence threshold | `0.3` |
| `--line-ratio` | Line position, `0` is top and `1` is bottom | `0.5` |
| `--line-config` | Path to a `line.json` file (overrides `--line-ratio`) | none |
| `--show` | Show a live preview window | off |
| `--imgsz` | Inference size. Larger sizes help with small or crowded people but are slower | `1280` |
| `--iou` | NMS IoU threshold. Lower it if overlapping people merge into one box | `0.5` |
| `--max-det` | Maximum detections per frame | `1000` |

Smaller models are faster and larger ones are more accurate. Larger models are slow on CPU, so test on your own machine first.
