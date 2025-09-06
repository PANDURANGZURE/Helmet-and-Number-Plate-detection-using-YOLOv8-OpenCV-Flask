import os, re, json, time, argparse, difflib
import cv2
import numpy as np
import easyocr
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# -------------------- Owners DB --------------------
def normalize_plate(s: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', s.upper())

if not os.path.exists("owners.json"):
    sample_data = {
        "ET0705": {"name": "Rahul Kumar", "phone": "9876543210"},
        "MH12AB1234": {"name": "Sneha Patil", "phone": "9123456780"}
    }
    with open("owners.json", "w") as f:
        json.dump(sample_data, f, indent=4)

with open("owners.json", "r") as f:
    owners_raw = json.load(f)

owners = {normalize_plate(k): v for k, v in owners_raw.items()}
owner_keys = list(owners.keys())

# -------------------- OCR Setup --------------------
reader = easyocr.Reader(['en'], gpu=False)
ALLOW = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$'),
    re.compile(r'^[A-Z]{2}\d{1}[A-Z]{1,2}\d{4}$'),
]

AMBIGUOUS_FIX = str.maketrans({
    'O': '0',  # sometimes O->0 (tune below)
    'I': '1',
    'L': '1',
    'S': '5',
    'B': '8',
    # We'll also try the reverse path in scoring
})

def is_plate_like(text: str) -> bool:
    t = normalize_plate(text)
    if len(t) < 6 or len(t) > 12:
        return False
    return any(p.fullmatch(t) for p in PLATE_PATTERNS)

def preprocess_for_ocr(bgr_roi):
    # robust preprocessing: grayscale -> CLAHE -> adaptive thresh
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    return thr

def try_fix_ambiguous(text: str) -> str:
    t = normalize_plate(text)
    # try both direction heuristics and keep the best pattern-like string
    candidates = set([t, t.translate(AMBIGUOUS_FIX)])
    # Also try replacing zeros that look like O in state code positions (first 2 chars are letters)
    if len(t) >= 2:
        t2 = list(t)
        for i in range(min(2, len(t2))):
            if t2[i] == '0':
                t2[i] = 'O'
        candidates.add(''.join(t2))
    # pick the candidate that matches plate pattern or longest length
    best = None
    for c in candidates:
        if is_plate_like(c):
            return c
        if best is None or len(c) > len(best):
            best = c
    return best or t

def ocr_best_plate(img_bgr):
    # Run OCR on both original ROI and preprocessed ROI and choose best by confidence & plate-likeness
    prep = preprocess_for_ocr(img_bgr)
    to_try = [
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(prep, cv2.COLOR_GRAY2RGB)
    ]
    best = None
    for img_rgb in to_try:
        results = reader.readtext(
            img_rgb,
            detail=1,
            paragraph=False,
            allowlist=ALLOW,
            decoder='greedy'
        )
        for (bbox, text, prob) in results:
            raw = normalize_plate(text)
            if not raw:
                continue
            fixed = try_fix_ambiguous(raw)
            score = prob * (1.3 if is_plate_like(fixed) else 1.0)
            if best is None or score > best[1]:
                best = (fixed, score, prob, bbox)
    if best:
        return best[0], best[2], best[3]
    return None, None, None

# -------------------- YOLO Model --------------------
model = YOLO("helnumbest.pt")  # your trained model

def get_plate_class_ids(model):
    """Return class ids that look like number plate classes."""
    names = getattr(model, "names", {})
    plate_ids = []
    for k, v in names.items():
        name = str(v).lower()
        if "plate" in name or "licen" in name:  # license/number plate
            plate_ids.append(int(k))
    # If nothing matched, fallback to [0] but log it
    if not plate_ids:
        print("[WARN] Could not find 'plate' class in model.names. Falling back to class 0.")
        plate_ids = [0]
    return set(plate_ids)

PLATE_CLASS_IDS = get_plate_class_ids(model)

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict_img():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    f = request.files['file']
    filename = secure_filename(f.filename)
    if not filename:
        return "Invalid filename", 400

    basepath = os.path.dirname(__file__)
    upload_dir = os.path.join(basepath, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, filename)
    f.save(filepath)

    # ------------------ Video Handling ------------------
    if filename.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return "Failed to open video", 500

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            res_plotted = results[0].plot()
            out.write(res_plotted)
        cap.release()
        out.release()
        return video_feed()
    
    return "Invalid file format. Upload only MP4 video.", 400

# -------------------- Video Streaming --------------------
def get_frame():
    video = cv2.VideoCapture("output.mp4")
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)  # ~30 fps

@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------- Capture + OCR + Owner Fetch --------------------
def iter_sampled_frames(video_path, samples=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # fallback: read first few frames
        count = 0
        while count < samples:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
            count += 1
        cap.release()
        return

    idxs = np.linspace(0, max(0, total-1), num=samples, dtype=int)
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            yield frame
    cap.release()

def best_owner_match(plate_norm: str, cutoff=0.8):
    """Exact match if available; otherwise fuzzy match to nearest owner key."""
    if plate_norm in owners:
        return owners[plate_norm], "exact", plate_norm
    # fuzzy
    close = difflib.get_close_matches(plate_norm, owner_keys, n=1, cutoff=cutoff)
    if close:
        return owners[close[0]], "fuzzy", close[0]
    return {"name": "Unknown", "phone": "N/A"}, "none", None

@app.route("/capture", methods=["POST"])
def capture():
    video_path = "output.mp4"
    if not os.path.exists(video_path):
        return jsonify({"error": "No processed video found. Upload first."}), 400

    detections = []
    seen = set()

    for frame in iter_sampled_frames(video_path, samples=5):
        results = model(frame)
        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in PLATE_CLASS_IDS:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            # pad a bit to not cut characters at the edge
            h, w = frame.shape[:2]
            pad = 4
            x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
            x2p = min(w, x2 + pad); y2p = min(h, y2 + pad)
            roi = frame[y1p:y2p, x1p:x2p]
            if roi.size == 0:
                continue

            plate_text, conf, _ = ocr_best_plate(roi)
            if not plate_text:
                continue

            plate_norm = normalize_plate(plate_text)
            if plate_norm in seen:
                continue
            seen.add(plate_norm)

            owner, match_type, matched_key = best_owner_match(plate_norm, cutoff=0.78)

            detections.append({
                "plate_detected": plate_norm,
                "matched_plate_key": matched_key or plate_norm,
                "match_type": match_type,                 # "exact", "fuzzy", or "none"
                "ocr_confidence": round(float(conf or 0), 3),
                "owner_name": owner.get("name", "Unknown"),
                "owner_phone": owner.get("phone", "N/A")
            })

    if not detections:
        return jsonify({"error": "No plate detected"}), 200

    # Deduplicate by matched plate key to avoid repeats across frames
    unique = {}
    for d in detections:
        key = d.get("matched_plate_key") or d.get("plate_detected")
        if key not in unique:
            unique[key] = d
        else:
            # keep the one with higher OCR confidence
            if d["ocr_confidence"] > unique[key]["ocr_confidence"]:
                unique[key] = d

    return jsonify(list(unique.values()))

# -------------------- Search by Plate API --------------------
@app.route("/get_owner", methods=["GET"])
def get_owner():
    plate = request.args.get("plate", "")
    plate_norm = normalize_plate(plate)
    if not plate_norm:
        return jsonify({"error": "Invalid plate format"}), 400
    
    owner, match_type, matched_key = best_owner_match(plate_norm, cutoff=0.78)
    return jsonify({
        "query_plate": plate_norm,
        "matched_plate_key": matched_key or plate_norm,
        "match_type": match_type,
        "owner_name": owner.get("name", "Unknown"),
        "owner_phone": owner.get("phone", "N/A")
    })

# -------------------- Run App --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int)
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
