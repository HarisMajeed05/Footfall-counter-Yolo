"""
footfall_counter.py
====================
Minimal, single-file people counter. Detect people (YOLO) + track them
(ByteTrack, built into ultralytics) + count line crossings (Entry/Exit).

FOLDER STRUCTURE THIS SCRIPT ASSUMES
-------------------------------------
project_root/
├── Models/              # model weight files (yolo11m.pt, rtdetr-l.pt, etc.)
├── Line_Config/          # saved line.json files from tools/select_line.py
├── Output/                # annotated output videos land here by default
├── Test/                  # your test videos
├── tools/
│   └── select_line.py
└── footfall_counter.py

USAGE
-----
    python footfall_counter.py --source Test/input.mp4 --show
    python footfall_counter.py --source Test/input.mp4 --output Output/result.mp4
    python footfall_counter.py --webcam 0
    python footfall_counter.py --source Test/input.mp4 --line-config Line_Config/line.json

MODEL NAMES
-----------
Just pass the short name — no path, no ".pt" needed:
    --model yolo11m        -> resolves to Models/yolo11m.pt
    --model yolo11n.pt      -> resolves to Models/yolo11n.pt (also fine with .pt included)
    --model rtdetr-l        -> resolves to Models/rtdetr-l.pt
If the file doesn't exist in Models/ yet, ultralytics downloads it there
automatically on first run.

FIRST STEP IF SOMETHING SEEMS WRONG
------------------------------------
Run with --show and watch the RAW DETECTIONS before worrying about the line
or the counting: are green boxes appearing on real people at all? If yes,
detection is working and any remaining issue is about the line position or
counting logic. If no boxes ever appear, the problem is upstream of
everything else (model, video file, or --conf too high) and no amount of
counting-logic fixing will matter until that's resolved.

CROWDED-SCENE TUNING
---------------------
If boxes are being missed specifically where people overlap in a dense
crowd (not everywhere), the usual cause is NOT the model choice — it's
three inference-time settings that default to values tuned for sparse
scenes:
  --imgsz    default 640 downscales the frame before inference, which
             squashes small/overlapping people together. Raise this
             (960/1280/1536) for dense crowds.
  --iou      default 0.7 NMS threshold. In a dense crowd, two genuinely
             different people can have high box overlap, and NMS will
             delete the lower-confidence one as a "duplicate". Raising
             this keeps more real, distinct, overlapping people.
  --max-det  default 300 detections per frame. Very dense frames can hit
             this cap and silently drop people. Raised as a safety net.
"""

import argparse
import json
import os

import cv2
from ultralytics import YOLO

MODELS_DIR = "Models"
OUTPUT_DIR = "Output"
LINE_CONFIG_DIR = "Line_Config"


def resolve_model_path(name: str) -> str:
    """Turn a short model name like 'yolo11m' or 'yolo11m.pt' into
    'Models/yolo11m.pt', regardless of which form the user typed."""
    filename = name if name.endswith(".pt") else f"{name}.pt"
    filename = os.path.basename(filename)  # strip any path the user accidentally included
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, filename)


def side_of_line(p1, p2, point):
    """Sign of cross product — tells you which side of line p1->p2 a point is on.
    Positive on one side, negative on the other, works for any line angle."""
    return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])


def select_line_interactively(cap, window_name="Footfall Counter"):
    """Grab live frames from an already-open webcam capture and let the user
    click two points to define the counting line, right in the --show window.

    Click point 1, click point 2, then press ENTER/SPACE to confirm.
    Press 'r' to reset and pick again. Press 'q' or ESC to abort.
    Returns (p1, p2) as (x, y) int tuples.
    """
    clicked = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 2:
            clicked.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print()
    print("Webcam line setup: click two points on the window to draw the counting line.")
    print("  Click point 1, then point 2.")
    print("  Press ENTER or SPACE to confirm, 'r' to reset, 'q'/ESC to abort.")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read a frame from the webcam for line setup.")

        display = frame.copy()
        for i, pt in enumerate(clicked):
            cv2.circle(display, pt, 6, (0, 0, 255), -1)
            cv2.putText(display, f"P{i + 1}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(clicked) == 2:
            cv2.line(display, clicked[0], clicked[1], (255, 0, 0), 2)
            cv2.putText(display, "Press ENTER/SPACE to confirm, 'r' to reset",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Click point {len(clicked) + 1}/2",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (13, 32) and len(clicked) == 2:  # ENTER or SPACE
            return clicked[0], clicked[1]
        elif key == ord("r"):
            clicked.clear()
        elif key in (27, ord("q")):  # ESC or 'q'
            raise RuntimeError("Line setup aborted by user.")


def main():
    ap = argparse.ArgumentParser(description="Minimal footfall counter")
    ap.add_argument("--source", default=None, help="Path to a video file, e.g. Test/input.mp4")
    ap.add_argument("--webcam", type=int, default=None, help="Webcam index, e.g. 0")
    ap.add_argument("--output", default=None,
                     help=f"Output video path. Defaults to {OUTPUT_DIR}/<source-name>_output.mp4")
    ap.add_argument("--model", default="yolo11m",
                     help=f"Model name only, e.g. 'yolo11m' or 'rtdetr-l' — automatically resolved "
                          f"to {MODELS_DIR}/<name>.pt. A .pt extension is also accepted.")
    ap.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    ap.add_argument("--line-ratio", type=float, default=0.5,
                     help="Horizontal counting line position, 0.0 (top) to 1.0 (bottom). "
                          "Ignored if --line-config is given.")
    ap.add_argument("--line-config", default=None,
                     help=f"Path to a line.json from tools/select_line.py, e.g. "
                          f"{LINE_CONFIG_DIR}/line.json, for a custom-angle line instead of the "
                          f"default horizontal one.")
    ap.add_argument("--show", action="store_true", help="Show a live window while processing")

    # --- Crowded-scene tuning (does not change any counting/drawing logic) ---
    ap.add_argument("--imgsz", type=int, default=1280,
                     help="Inference resolution. Higher = better recall on small/overlapping "
                          "people in dense crowds, but slower. Default 1280 (was implicitly 640).")
    ap.add_argument("--iou", type=float, default=0.5,
                     help="NMS IoU threshold. Raise this if boxes are being merged/deleted where "
                          "people overlap in a crowd.")
    ap.add_argument("--max-det", type=int, default=1000,
                     help="Max detections per frame. Raised from the library default of 300 so "
                          "dense-crowd frames don't silently hit the cap.")

    args = ap.parse_args()

    if args.webcam is not None:
        source = args.webcam
        args.show = True
        source_name = f"webcam{args.webcam}"
    elif args.source:
        source = args.source
        source_name = os.path.splitext(os.path.basename(args.source))[0]
    else:
        ap.error("Provide --source <video path> or --webcam <index>")

    model_path = resolve_model_path(args.model)
    print(f"Loading model: {model_path} "
          f"({'found locally' if os.path.exists(model_path) else 'will download'})")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.line_config:
        with open(args.line_config) as f:
            cfg = json.load(f)
        line_p1, line_p2 = tuple(cfg["p1"]), tuple(cfg["p2"])
        print(f"Using custom line from {args.line_config}: {line_p1} -> {line_p2}")
    elif args.webcam is not None:
        # No --line-config for a live webcam: ask the user to draw the line
        # right in the --show window before any detection/tracking starts.
        line_p1, line_p2 = select_line_interactively(cap)
        print(f"Webcam line set interactively: {line_p1} -> {line_p2}")
    else:
        line_y = int(height * args.line_ratio)
        line_p1, line_p2 = (0, line_y), (width, line_y)
        print(f"Using default horizontal line at y={line_y} "
              f"(pass --line-config for a custom-angle line)")

    output_path = args.output or os.path.join(OUTPUT_DIR, f"{source_name}_output.mp4")

    print(f"Source opened OK: {width}x{height} @ {fps:.1f}fps")
    print(f"Model: {model_path} | conf={args.conf} | imgsz={args.imgsz} | "
          f"iou={args.iou} | max_det={args.max_det}")
    print(f"Output will be saved to: {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if args.show:
        cv2.namedWindow("Footfall Counter", cv2.WINDOW_NORMAL)
        scale = min(1.0, 1000 / max(width, height))
        cv2.resizeWindow("Footfall Counter", int(width * scale), int(height * scale))

    prev_side = {}        # track_id -> last side-of-line value (+/-) of its center point
    counted_entry = set()
    counted_exit = set()
    entry_count = 0
    exit_count = 0
    frame_num = 0
    total_detections_seen = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1

        # Run detection + tracking on THIS frame, nothing skipped, nothing cached.
        result = model.track(
            frame,
            conf=args.conf,
            classes=[0],          # COCO class 0 = person
            tracker="bytetrack.yaml",
            persist=True,
            device="cpu",
            verbose=False,
            imgsz=args.imgsz,     # higher res so overlapping/small people don't collapse together
            iou=args.iou,         # NMS threshold tuned for dense-crowd overlap
            max_det=args.max_det, # safety net so dense frames don't hit the detection cap
        )[0]

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            total_detections_seen += len(ids)

            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Draw every raw detection, no filtering, no conditions.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                # Line-crossing count — works for the line at any angle
                s = side_of_line(line_p1, line_p2, (cx, cy))
                last_s = prev_side.get(tid)
                prev_side[tid] = s
                if last_s is not None and last_s != 0 and s != 0:
                    if last_s < 0 <= s and tid not in counted_entry:
                        entry_count += 1
                        counted_entry.add(tid)
                        counted_exit.discard(tid)
                    elif last_s > 0 >= s and tid not in counted_exit:
                        exit_count += 1
                        counted_exit.add(tid)
                        counted_entry.discard(tid)

        # Draw the counting line and stats every frame
        cv2.line(frame, line_p1, line_p2, (255, 0, 0), 2)
        cv2.rectangle(frame, (0, 0), (260, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"ENTRY: {entry_count}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"EXIT : {exit_count}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        writer.write(frame)

        if args.show:
            cv2.imshow("Footfall Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_num % 30 == 0:
            print(f"Frame {frame_num} | Entry={entry_count} Exit={exit_count} "
                  f"| total detections so far={total_detections_seen}")

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print()
    print(f"Done. Frames processed: {frame_num}")
    print(f"Total person detections across the whole video: {total_detections_seen}")
    print(f"Total ENTRY: {entry_count}")
    print(f"Total EXIT : {exit_count}")
    print(f"Output saved to: {output_path}")

    if total_detections_seen == 0:
        print()
        print("WARNING: zero people were detected in the entire video. This means the")
        print("problem is upstream of counting logic. Things to check, in order:")
        print("  1. Open the source video in a normal player — does it actually show people?")
        print("  2. Try a much lower --conf, e.g. --conf 0.1, to rule out threshold issues.")
        print("  3. Try --model yolo11n (smallest/fastest) as a sanity check that the")
        print("     model itself loads and runs correctly on your machine.")
        print("  4. Confirm the video isn't corrupted: 'ffprobe input.mp4' should show valid")
        print("     stream info without errors.")


if __name__ == "__main__":
    main()
