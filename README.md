# Footfall Counter — Fresh Minimal Version

One file. No frame skipping, no coasting, no filters, no adaptive anything.
Every frame is processed exactly as detected — this is the simplest version
that can work, so if something's wrong, it's actually findable.

## Setup

```bat
conda env create -f environment.yml
conda activate footfall-fresh
```

## Run

```bat
python footfall_counter.py --source Test/input.mp4 --model yolo11m --show
```

Webcam:
```bat
python footfall_counter.py --webcam 0 --model yolo11m
```

Save-only, no live window (a bit faster):
```bat
python footfall_counter.py --source Test/input.mp4 --model yolo11m --output Output/result.mp4
```

## If nothing is being detected/highlighted

Run with `--show` first. The script prints a running total of detections to
the terminal every 30 frames, and a diagnostic block at the end if the total
is zero. Work through it in this order — don't touch the line/counting logic
until step 1 passes:

1. **Are green boxes appearing on real people at all**, regardless of the
   blue line or Entry/Exit numbers? If yes, detection works — any remaining
   problem is specifically about line placement or crossing logic, not
   detection. If no boxes ever appear, stop here — everything below depends
   on this working first.
2. Lower the threshold to rule out it being too strict:
   ```bat
   python footfall_counter.py --source Test/input.mp4 --model yolo11m --show --conf 0.1
   ```
3. Confirm the model itself loads and runs on your machine with the
   smallest, most reliable model:
   ```bat
   python footfall_counter.py --source Test/input.mp4 --show --model yolo11n
   ```
4. Confirm the video file isn't the problem — open it in a normal media
   player. If it won't play there either, it's not a code issue.

## Counting line

By default it's a straight horizontal line at 50% frame height (`--line-ratio`
moves it up/down). For a custom-angle line — click 2 points anywhere on the
frame — draw it once, then reuse it:

```bat
python tools/select_line.py --source Test/input.mp4
python footfall_counter.py --source Test/input.mp4 --model yolo11m --show --line-config Line_Config/line.json
```

`select_line.py` handles videos bigger than your screen correctly — the
preview is scaled down but clicks are mapped back to full resolution before
saving, so the line lands where you actually clicked.

## Flags

| Flag | Meaning | Default |
|---|---|---|
| `--source` | video file path | — |
| `--webcam` | webcam index instead of `--source` | — |
| `--output` | output video path | `output.mp4` |
| `--model` | YOLO weights file | `yolo11m.pt` |
| `--conf` | detection confidence threshold | `0.3` |
| `--line-ratio` | counting line height, 0=top 1=bottom (default horizontal line) | `0.5` |
| `--line-config` | path to a line.json from `tools/select_line.py`, for a custom-angle line | none |
| `--show` | live preview window | off |

## Once this works

Only after you've confirmed detection + counting are behaving correctly on
your actual footage should any additional feature (custom line angle, speed
optimization, false-positive filtering, etc.) go back in — one at a time,
tested against your real video after each change, not all at once.
