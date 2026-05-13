"""
DMS data-collection recorder — Tkinter GUI.

Captures the IMX477 CSI camera on sensor 0 (settings from config.yaml) and
saves a timestamped MP4 to recordings/. Built for in-lab data collection
during Assetto Corsa sessions.

Controls: Start/Stop button on the right panel. Window close (X) or Quit
button exits cleanly.
"""

import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

import cv2
import yaml
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inference.camera import Camera

PREVIEW_MAX = (960, 540)


def _load_camera_cfg() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)["camera"]


class RecorderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DMS Recorder")

        self.cfg = _load_camera_cfg()
        self.cam = Camera(self.cfg)

        self.writer = None
        self.rec_path: Path | None = None
        self.rec_start = 0.0
        self.frame_size = None
        self._photo = None  # keep a reference so Tk doesn't GC the image

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.after(10, self._tick)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        self.preview = ttk.Label(self.root, background="black")
        self.preview.grid(row=0, column=0, rowspan=8, sticky="nsew", **pad)

        side = ttk.Frame(self.root)
        side.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        ttk.Label(side, text="Status").grid(row=0, column=0, sticky="w")
        self.status_var = tk.StringVar(value="IDLE")
        self.status_lbl = ttk.Label(
            side, textvariable=self.status_var, font=("Helvetica", 18, "bold")
        )
        self.status_lbl.grid(row=1, column=0, sticky="w", pady=(0, 12))

        ttk.Label(side, text="Elapsed").grid(row=2, column=0, sticky="w")
        self.elapsed_var = tk.StringVar(value="00:00")
        ttk.Label(side, textvariable=self.elapsed_var,
                  font=("Helvetica", 14)).grid(row=3, column=0, sticky="w",
                                               pady=(0, 12))

        ttk.Label(side, text="Recording to").grid(row=4, column=0, sticky="w")
        self.path_var = tk.StringVar(value="—")
        ttk.Label(side, textvariable=self.path_var,
                  wraplength=260).grid(row=5, column=0, sticky="w",
                                       pady=(0, 12))

        ttk.Label(side, text="Last saved").grid(row=6, column=0, sticky="w")
        self.last_var = tk.StringVar(value="—")
        ttk.Label(side, textvariable=self.last_var,
                  wraplength=260).grid(row=7, column=0, sticky="w",
                                       pady=(0, 18))

        self.rec_btn = ttk.Button(side, text="Start Recording",
                                  command=self.toggle_record)
        self.rec_btn.grid(row=8, column=0, sticky="ew", pady=4)

        ttk.Button(side, text="Quit",
                   command=self.quit).grid(row=9, column=0, sticky="ew", pady=4)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _tick(self):
        ok, frame = self.cam.read()
        if not ok:
            self.status_var.set("CAMERA ERROR")
            self.root.after(100, self._tick)
            return

        self.frame_size = frame.shape[:2]  # (h, w)

        if self.writer is not None:
            self.writer.write(frame)
            elapsed = int(time.time() - self.rec_start)
            mm, ss = divmod(elapsed, 60)
            self.elapsed_var.set(f"{mm:02d}:{ss:02d}")

        display = frame
        if self.writer is not None:
            display = frame.copy()
            cv2.circle(display, (28, 32), 12, (0, 0, 255), -1)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail(PREVIEW_MAX)
        self._photo = ImageTk.PhotoImage(image=img)
        self.preview.configure(image=self._photo)

        if self.writer is not None:
            self.status_var.set("● REC")
            self.status_lbl.configure(foreground="#cc0000")
        else:
            self.status_var.set("IDLE")
            self.status_lbl.configure(foreground="#222222")

        self.root.after(1, self._tick)

    def toggle_record(self):
        if self.writer is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self.frame_size is None:
            return  # camera hasn't produced a frame yet
        out_dir = REPO_ROOT / "recordings"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"rec_{ts}.mp4"
        h, w = self.frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.cfg["framerate"], (w, h))
        if not writer.isOpened():
            self.path_var.set(f"ERROR: could not open writer\n{path}")
            return
        self.writer = writer
        self.rec_path = path
        self.rec_start = time.time()
        self.path_var.set(path.name)
        self.rec_btn.configure(text="Stop Recording")
        print(f"[REC] start -> {path}")

    def _stop_recording(self):
        if self.writer is None:
            return
        self.writer.release()
        saved = self.rec_path
        self.writer = None
        self.rec_path = None
        self.last_var.set(saved.name if saved else "—")
        self.path_var.set("—")
        self.elapsed_var.set("00:00")
        self.rec_btn.configure(text="Start Recording")
        print(f"[REC] stop  -> {saved}")

    def quit(self):
        if self.writer is not None:
            self._stop_recording()
        self.cam.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
