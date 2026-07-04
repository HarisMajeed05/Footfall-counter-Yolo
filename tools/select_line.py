"""
tools/select_line.py
=====================
Opens the first frame of a video and lets you click 2 points to draw a
counting line at ANY angle. Saves it to Line_Config/line.json by default,
which footfall_counter.py loads with --line-config.

Handles videos larger than your screen: the preview is scaled down to fit,
but clicks are mapped back to full video resolution before saving.

Controls: left-click 2 points, 's' to save, 'u' to undo, 'q' to cancel.

Usage:
    python tools/select_line.py --source Test/input.mp4
    python tools/select_line.py --source Test/input.mp4 --out Line_Config/my_line.json
"""

import argparse
import json
import os

import cv2

LINE_CONFIG_DIR = "Line_Config"

points = []          # in DISPLAY coordinates, gets mapped to original on save
scale = 1.0
base_frame = None
display_frame = None


def redraw():
    global display_frame
    display_frame = base_frame.copy()
    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(display_frame, "Click 2 points | s=save  u=undo  q=cancel",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for p in points:
        cv2.circle(display_frame, tuple(p), 6, (0, 255, 255), -1)
    if len(points) == 2:
        cv2.line(display_frame, tuple(points[0]), tuple(points[1]), (255, 120, 0), 3)


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append([x, y])
        redraw()


def main():
    global base_frame, scale

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="e.g. Test/input.mp4")
    ap.add_argument("--out", default=os.path.join(LINE_CONFIG_DIR, "line.json"),
                     help=f"Where to save the line. Defaults into {LINE_CONFIG_DIR}/")
    ap.add_argument("--max-dim", type=int, default=1000,
                     help="Max preview window size in pixels")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {args.source}")

    h, w = frame.shape[:2]
    scale = min(1.0, args.max_dim / max(w, h))
    base_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame.copy()
    if scale < 1.0:
        print(f"Video is {w}x{h}, showing scaled to fit your screen. "
              f"Clicks are mapped back to full resolution automatically.")

    redraw()
    window = "Draw counting line"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, base_frame.shape[1], base_frame.shape[0])
    cv2.setMouseCallback(window, on_click)

    print("Click 2 points to place the line, then press 's' to save.")
    print(f"Will save to: {args.out}")

    while True:
        cv2.imshow(window, display_frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            print("Cancelled, nothing saved.")
            break
        elif key == ord("u"):
            if points:
                points.pop()
                redraw()
        elif key == ord("s"):
            if len(points) != 2:
                print("Click exactly 2 points first.")
                continue
            p1 = [int(points[0][0] / scale), int(points[0][1] / scale)]
            p2 = [int(points[1][0] / scale), int(points[1][1] / scale)]
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"p1": p1, "p2": p2}, f, indent=2)
            print(f"Saved line {p1} -> {p2} to {args.out} (full-resolution coordinates)")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
