"""
STEP 1 — Extract hand landmarks → gestures.csv
===============================================
Your dataset has:
  - word_dataset/<gesture>/  *.jpg          (clean 640×480 hand images)
  - word_dataset/images/train/  *.xml       (bounding boxes per image)

Strategy:
  1. Parse bounding box from XML → crop the hand region from the image
  2. Run MediaPipe on the CROPPED hand (much better detection rate)
  3. Normalise the 21 landmarks → 42 floats
  4. Save to gestures.csv

Run:
    python extract.py
"""

import os, csv, xml.etree.ElementTree as ET
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATASET_DIR = r"C:\Users\Lenovo\Desktop\study\SEM IV\LA\word_dataset"
XML_DIR     = r"C:\Users\Lenovo\Desktop\study\SEM IV\LA\word_datasetimages\train"
OUTPUT_CSV  = r"C:\Users\Lenovo\Desktop\study\SEM IV\LA\word_dataset\gestures.csv"
MP_MODEL    = r"C:\Users\Lenovo\Desktop\study\SEM IV\LA\word_dataset\hand_landmarker.task"
PAD         = 30     # pixels of padding around bounding box crop
# ──────────────────────────────────────────────────────────────────────────────


def download_model():
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
    print(f"Downloading MediaPipe hand model (~25 MB) → {MP_MODEL}")
    urllib.request.urlretrieve(url, MP_MODEL)
    print("Done.\n")


def load_bboxes(xml_dir):
    """
    Returns dict: image_filename (no ext) → (xmin, ymin, xmax, ymax)
    """
    bboxes = {}
    if not os.path.exists(xml_dir):
        print(f"  WARNING: XML dir not found: {xml_dir}")
        return bboxes

    for f in os.listdir(xml_dir):
        if not f.endswith('.xml'):
            continue
        try:
            tree  = ET.parse(os.path.join(xml_dir, f))
            root  = tree.getroot()
            bb    = root.find('.//bndbox')
            xmin  = int(float(bb.find('xmin').text))
            ymin  = int(float(bb.find('ymin').text))
            xmax  = int(float(bb.find('xmax').text))
            ymax  = int(float(bb.find('ymax').text))
            fname = os.path.splitext(f)[0]  # e.g. "hello.c003..."
            bboxes[fname] = (xmin, ymin, xmax, ymax)
        except Exception:
            pass
    return bboxes


def normalise(landmarks):
    """
    Position- and scale-independent 42-float vector.
    Centre on wrist (pt 0), scale by wrist→mid-MCP distance.
    """
    pts   = np.array([[lm.x, lm.y] for lm in landmarks])
    pts  -= pts[0].copy()
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts  /= scale
    return pts.flatten()


def detect_on_image(detector, img_bgr, bbox=None, pad=PAD):
    """
    Try detection on full image first.
    If bbox given, also try on cropped region.
    Returns landmarks or None.
    """
    h, w = img_bgr.shape[:2]

    def run(bgr):
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    # 1. try full image
    lms = run(img_bgr)
    if lms:
        return lms

    # 2. try cropped using bbox
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        x1 = max(0, xmin - pad)
        y1 = max(0, ymin - pad)
        x2 = min(w, xmax + pad)
        y2 = min(h, ymax + pad)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            lms = run(crop)
            if lms:
                return lms

    # 3. try resized (sometimes helps with small hands)
    small = cv2.resize(img_bgr, (320, 240))
    lms   = run(small)
    return lms


def main():
    if not os.path.exists(MP_MODEL):
        download_model()

    bboxes = load_bboxes(XML_DIR)
    print(f"Loaded {len(bboxes)} bounding boxes from XML files.\n")

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MP_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )

    gesture_folders = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
        and d not in ('images',)       # skip the images/train XML folder
    ])
    print(f"Gesture folders: {gesture_folders}\n")

    rows     = []
    total_ok = 0
    total_sk = 0

    with HandLandmarker.create_from_options(options) as detector:
        for label in gesture_folders:
            folder = os.path.join(DATASET_DIR, label)
            images = [f for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
            ok = 0

            for img_file in images:
                img_path = os.path.join(folder, img_file)
                img_bgr  = cv2.imread(img_path)
                if img_bgr is None:
                    total_sk += 1
                    continue

                # look up bounding box for this image
                key   = os.path.splitext(img_file)[0]
                bbox  = bboxes.get(key)

                try:
                    lms = detect_on_image(detector, img_bgr, bbox)
                    if lms:
                        vec = normalise(lms)
                        rows.append([label] + vec.tolist())
                        ok += 1
                    else:
                        total_sk += 1
                except Exception as e:
                    print(f"    Error on {img_file}: {e}")
                    total_sk += 1

            total_ok += ok
            status = "" if ok > 0 else "  ← WARNING: 0 detected"
            print(f"  [{label:15s}]  {ok:3d}/{len(images):3d} extracted{status}")

    if total_ok == 0:
        print("\n" + "="*60)
        print("0 landmarks extracted. Possible causes:")
        print("  1. Images have pre-drawn skeleton overlay (red dots)")
        print("     → find the clean image version in your dataset")
        print("  2. Hand is very small or partially out of frame")
        print("  3. MediaPipe model version mismatch")
        print("="*60)
        return

    # write CSV
    header = ['label'] + [f'{ax}{i}' for i in range(21) for ax in ('x','y')]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✓  {total_ok} vectors saved → {OUTPUT_CSV}")
    print(f"   Skipped (no detection): {total_sk}")
    print(f"\nGesture counts in CSV:")
    from collections import Counter
    counts = Counter(r[0] for r in rows)
    for g, c in sorted(counts.items()):
        print(f"  {g:15s}: {c}")
    print("\nNext: run train_pca.py")


if __name__ == "__main__":
    main()
