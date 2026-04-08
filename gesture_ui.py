"""
ASL Real-Time Gesture Recognition — Full GUI
=============================================
Opens your webcam inside a clean Tkinter window.
Detects your hand every frame, projects the 42-dim vector
onto the PCA subspaces, and displays the gesture meaning live.

Run:
    python gesture_ui.py

Requirements:
    pip install mediapipe opencv-python numpy pillow

Files needed in the same folder:
    hand_landmarker.task   (auto-downloaded on first run)
    pca_model.npz          (created by train_pca.py)
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import font as tkfont
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
CAMERA_ID      = 0                              # change to 1 if wrong camera
MODEL_NPZ      = r"C:\Users\Lenovo\Desktop\study\SEM IV\LA\pca_model.npz"
MP_MODEL       = r"C:\Users\Lenovo\Desktop\study\SEM IV\LA\word_dataset\hand_landmarker.task"
THRESHOLD      = 0.40                           # max distance to show label
FRAME_W        = 720
FRAME_H        = 540
# ──────────────────────────────────────────────────────────────────────────────

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def download_mp_model(path):
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print(f"Downloading MediaPipe hand model (~25 MB) → {path}")
    urllib.request.urlretrieve(url, path)
    print("Download complete.\n")


def load_pca_model(path):
    data        = np.load(path, allow_pickle=True)
    class_names = data['class_names'].tolist()
    model = {}
    for lbl in class_names:
        s = lbl.replace(' ', '_')
        model[lbl] = {
            'mean'     : data[f"{s}__mean"],
            'eigenvecs': data[f"{s}__eigenvecs"],
        }
    return model, class_names


def normalise(landmarks):
    pts   = np.array([[lm.x, lm.y] for lm in landmarks])
    pts  -= pts[0].copy()
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts  /= scale
    return pts.flatten()


def classify(vec, model, class_names):
    dists = {}
    for lbl in class_names:
        mu  = model[lbl]['mean']
        U   = model[lbl]['eigenvecs']
        vc  = vec - mu
        d   = float(np.linalg.norm(vc - U @ (U.T @ vc)))
        dists[lbl] = d
    best = min(dists, key=dists.get)
    return best, dists[best], dists


def draw_skeleton_on_frame(frame, landmarks):
    """Draw white skeleton + green dots on an OpenCV BGR frame."""
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 7 if i == 0 else 5
        cv2.circle(frame, (x, y), r, (0, 220, 110), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (0, 0, 0), 1,  cv2.LINE_AA)


# ── main application ───────────────────────────────────────────────────────────

class ASLApp:
    def __init__(self, root):
        self.root     = root
        self.running  = True

        # ── state ──────────────────────────────────────────────────────────
        self.label      = "—"
        self.dist       = 999.0
        self.all_dists  = {}
        self.hand_found = False
        self.fps        = 0.0
        self.threshold  = THRESHOLD
        self.frame_rgb  = None
        self.lock       = threading.Lock()

        # ── load models ────────────────────────────────────────────────────
        if not os.path.exists(MP_MODEL):
            download_mp_model(MP_MODEL)

        self.pca_model, self.class_names = load_pca_model(MODEL_NPZ)
        print(f"Gestures loaded: {self.class_names}\n")

        # ── build UI ───────────────────────────────────────────────────────
        self._build_ui()

        # ── start camera thread ────────────────────────────────────────────
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = HandLandmarker.create_from_options(options)

        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

        # ── start UI refresh loop ──────────────────────────────────────────
        self.root.after(30, self._refresh_ui)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("ASL Recognition — PCA Subspace")
        self.root.configure(bg="#0f0f14")
        self.root.resizable(False, False)

        # top bar
        top = tk.Frame(self.root, bg="#0f0f14", pady=8)
        top.pack(fill="x")
        tk.Label(top, text="ASL Real-Time Recognition",
                 bg="#0f0f14", fg="#e8e6df",
                 font=("Helvetica", 15, "bold")).pack(side="left", padx=16)
        self.fps_lbl = tk.Label(top, text="FPS: —",
                                bg="#0f0f14", fg="#555",
                                font=("Courier", 11))
        self.fps_lbl.pack(side="right", padx=16)

        # main area: video + side panel
        main = tk.Frame(self.root, bg="#0f0f14")
        main.pack(fill="both", expand=True, padx=12, pady=4)

        # video canvas
        self.canvas = tk.Canvas(main, width=FRAME_W, height=FRAME_H,
                                bg="#1a1a22", highlightthickness=0)
        self.canvas.pack(side="left")

        # side panel
        side = tk.Frame(main, bg="#13151c", width=220,
                        padx=14, pady=14)
        side.pack(side="left", fill="y", padx=(10, 0))
        side.pack_propagate(False)

        tk.Label(side, text="DETECTED GESTURE",
                 bg="#13151c", fg="#555",
                 font=("Courier", 9)).pack(anchor="w")

        self.big_label = tk.Label(side, text="—",
                                  bg="#13151c", fg="#2ec99e",
                                  font=("Helvetica", 28, "bold"),
                                  wraplength=190, justify="left")
        self.big_label.pack(anchor="w", pady=(4, 0))

        self.conf_label = tk.Label(side, text="No hand",
                                   bg="#13151c", fg="#555",
                                   font=("Helvetica", 10))
        self.conf_label.pack(anchor="w", pady=(2, 12))

        tk.Frame(side, bg="#222", height=1).pack(fill="x", pady=6)

        tk.Label(side, text="ALL DISTANCES",
                 bg="#13151c", fg="#555",
                 font=("Courier", 9)).pack(anchor="w")

        # distance bars container
        self.bars_frame = tk.Frame(side, bg="#13151c")
        self.bars_frame.pack(fill="x", pady=(6, 12))
        self._bar_widgets = {}   # label → (name_lbl, bar_canvas, dist_lbl)

        tk.Frame(side, bg="#222", height=1).pack(fill="x", pady=6)

        # threshold control
        tk.Label(side, text="CONFIDENCE THRESHOLD",
                 bg="#13151c", fg="#555",
                 font=("Courier", 9)).pack(anchor="w")
        self.thresh_val_lbl = tk.Label(side,
                                       text=f"{self.threshold:.2f}",
                                       bg="#13151c", fg="#e8e6df",
                                       font=("Courier", 11))
        self.thresh_val_lbl.pack(anchor="w")

        self.thresh_slider = tk.Scale(
            side, from_=0.10, to=1.00, resolution=0.01,
            orient="horizontal", length=190,
            bg="#13151c", fg="#e8e6df", troughcolor="#222",
            highlightthickness=0, bd=0,
            command=self._on_threshold_change
        )
        self.thresh_slider.set(self.threshold)
        self.thresh_slider.pack(fill="x", pady=(2, 12))

        # screenshot button
        btn = tk.Button(side, text="Save Screenshot",
                        bg="#222", fg="#e8e6df",
                        relief="flat", padx=8, pady=6,
                        font=("Helvetica", 10),
                        command=self._screenshot)
        btn.pack(fill="x", pady=(4, 0))

        # hint
        tk.Label(side, text="Lower threshold = stricter",
                 bg="#13151c", fg="#444",
                 font=("Courier", 8)).pack(anchor="w", pady=(8, 0))

    def _make_bar_row(self, lbl):
        """Create one distance-bar row in the side panel."""
        row = tk.Frame(self.bars_frame, bg="#13151c")
        row.pack(fill="x", pady=2)

        name = tk.Label(row, text=lbl[:10], width=10, anchor="w",
                        bg="#13151c", fg="#aaa", font=("Courier", 9))
        name.pack(side="left")

        bar_canvas = tk.Canvas(row, height=10, width=100,
                               bg="#1e1e28", highlightthickness=0)
        bar_canvas.pack(side="left", padx=(2, 4))

        dist_lbl = tk.Label(row, text="—", width=6, anchor="e",
                            bg="#13151c", fg="#555", font=("Courier", 8))
        dist_lbl.pack(side="left")

        self._bar_widgets[lbl] = (name, bar_canvas, dist_lbl)

    # ── camera loop (runs in background thread) ────────────────────────────────

    def _camera_loop(self):
        prev_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)   # mirror
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_img)

            hand = bool(result.hand_landmarks)
            lbl, dist, dists = "—", 999.0, {}

            if hand:
                lms = result.hand_landmarks[0]
                draw_skeleton_on_frame(frame, lms)
                vec = normalise(lms)
                lbl, dist, dists = classify(vec, self.pca_model, self.class_names)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # re-convert after drawing

            # FPS
            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            with self.lock:
                self.frame_rgb  = rgb
                self.hand_found = hand
                self.label      = lbl
                self.dist       = dist
                self.all_dists  = dists
                self.fps        = fps

    # ── UI refresh (runs on main thread via after()) ───────────────────────────

    def _refresh_ui(self):
        if not self.running:
            return

        with self.lock:
            frame_rgb  = self.frame_rgb
            hand_found = self.hand_found
            label      = self.label
            dist       = self.dist
            all_dists  = dict(self.all_dists)
            fps        = self.fps

        # ── update video canvas ────────────────────────────────────────────
        if frame_rgb is not None:
            img = Image.fromarray(frame_rgb)
            img = img.resize((FRAME_W, FRAME_H), Image.BILINEAR)

            # draw bottom-left overlay directly on PIL image
            self._draw_frame_overlay(img, label, dist, hand_found)

            imgtk = ImageTk.PhotoImage(img)
            self.canvas.imgtk = imgtk          # hold reference
            self.canvas.create_image(0, 0, anchor="nw", image=imgtk)

        # ── update side panel ──────────────────────────────────────────────
        self.fps_lbl.config(text=f"FPS: {fps:.0f}")

        confident = hand_found and dist < self.threshold

        if not hand_found:
            self.big_label.config(text="—", fg="#555")
            self.conf_label.config(text="Show your hand...", fg="#555")
        elif confident:
            self.big_label.config(text=label.upper(), fg="#2ec99e")
            self.conf_label.config(text=f"dist {dist:.3f}  ✓ confident", fg="#2ec99e")
        else:
            self.big_label.config(text=label.upper(), fg="#888")
            self.conf_label.config(text=f"dist {dist:.3f}  low confidence", fg="#f5a623")

        # ── distance bars ──────────────────────────────────────────────────
        if all_dists:
            max_d = max(all_dists.values()) + 1e-6
            # create rows if first time
            for lbl_name in self.class_names:
                if lbl_name not in self._bar_widgets:
                    self._make_bar_row(lbl_name)

            for lbl_name, d in all_dists.items():
                if lbl_name not in self._bar_widgets:
                    self._make_bar_row(lbl_name)
                _, bar_c, dist_l = self._bar_widgets[lbl_name]
                bar_w = int((d / max_d) * 100)
                bar_c.delete("all")
                is_best = (lbl_name == label) and confident
                color   = "#2ec99e" if is_best else "#334"
                bar_c.create_rectangle(0, 0, bar_w, 10, fill=color, outline="")
                dist_l.config(text=f"{d:.3f}")

        self.root.after(30, self._refresh_ui)

    # ── frame overlay (drawn on PIL image) ────────────────────────────────────

    def _draw_frame_overlay(self, img, label, dist, hand_found):
        """Draw a semi-transparent prediction box in the bottom-left corner."""
        draw = ImageDraw.Draw(img, "RGBA")
        confident = hand_found and dist < self.threshold

        # background panel
        bx, by = 14, FRAME_H - 110
        draw.rectangle([(bx, by), (bx + 280, by + 96)],
                       fill=(15, 15, 20, 190))

        # accent stripe
        stripe_color = (46, 201, 158, 255) if confident else (60, 60, 60, 255)
        draw.rectangle([(bx, by), (bx + 5, by + 96)], fill=stripe_color)

        # label text — use default font (no path needed)
        text    = label.upper() if (hand_found and confident) else ("— — —" if not hand_found else label.upper())
        t_color = (46, 201, 158, 255) if confident else (120, 120, 120, 255) if hand_found else (70, 70, 70, 255)

        try:
            fnt_big  = ImageFont.truetype("arial.ttf", 32)
            fnt_small= ImageFont.truetype("arial.ttf", 12)
        except Exception:
            fnt_big  = ImageFont.load_default()
            fnt_small= ImageFont.load_default()

        draw.text((bx + 14, by + 12), text, font=fnt_big, fill=t_color)
        draw.text((bx + 14, by + 56), f"dist: {dist:.3f}   thresh: {self.threshold:.2f}",
                  font=fnt_small, fill=(160, 160, 160, 200))

        status = "CONFIDENT" if confident else ("No hand" if not hand_found else "Low confidence")
        s_color = (46, 201, 158, 200) if confident else (100, 100, 100, 180)
        draw.text((bx + 14, by + 74), status, font=fnt_small, fill=s_color)

    # ── controls ───────────────────────────────────────────────────────────────

    def _on_threshold_change(self, val):
        self.threshold = float(val)
        self.thresh_val_lbl.config(text=f"{self.threshold:.2f}")

    def _screenshot(self):
        with self.lock:
            frame = self.frame_rgb
        if frame is None:
            return
        path = os.path.join(r"D:\AI\Hand_symbol",
                            f"screenshot_{int(time.time())}.png")
        Image.fromarray(frame).save(path)
        print(f"Screenshot saved → {path}")

    def _on_close(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()
        self.detector.close()
        self.root.destroy()


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # sanity checks
    if not os.path.exists(MODEL_NPZ):
        print(f"ERROR: PCA model not found at:\n  {MODEL_NPZ}")
        print("Run train_pca.py first to generate it.")
        sys.exit(1)

    if not os.path.exists(MP_MODEL):
        print(f"MediaPipe model not found at:\n  {MP_MODEL}")
        print("It will be downloaded automatically when the app starts.\n")

    root = tk.Tk()
    app  = ASLApp(root)
    root.mainloop()
