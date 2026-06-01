

# ── 0. CÀI ĐẶT ──────────────────────────────────────────────
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "ultralytics", "easyocr", "-q"], check=True)

import os, glob, shutil, re, time, yaml, zipfile
import cv2, numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import torch
from ultralytics import YOLO
import easyocr
from google.colab import files

# ── 1. KIỂM TRA GPU ─────────────────────────────────────────
print("=" * 60)
print("  LICENSE PLATE RECOGNITION — YOLOv8s + EasyOCR  [v3]")
print("=" * 60)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("\n⚠️  Không có GPU — Runtime > Change runtime type > T4 GPU")
print()

# ── 2. UPLOAD FILE ZIP ──────────────────────────────────────
# Dataset dùng: archive__2_.zip  (KHÔNG cần Kaggle API)
# Cấu trúc bên trong zip:
#   YOLO_dataset/images/train/*.png
#   YOLO_dataset/images/val/*.png
#   YOLO_dataset/labels/train/*.txt
#   YOLO_dataset/labels/val/*.txt
#   YOLO_dataset/data.yaml

print("📂 Upload file archive__2_.zip (dataset của bạn):")
uploaded = files.upload()
zip_filename = list(uploaded.keys())[0]
print(f"✅ Đã nhận file: {zip_filename}\n")

# ── 3. GIẢI NÉN ─────────────────────────────────────────────
EXTRACT_DIR = "/content/raw_dataset"
os.makedirs(EXTRACT_DIR, exist_ok=True)

print(f"📦 Đang giải nén vào {EXTRACT_DIR} ...")
with zipfile.ZipFile(zip_filename, "r") as zf:
    zf.extractall(EXTRACT_DIR)
print("✅ Giải nén xong\n")

# ── 4. TỰ ĐỘNG XÁC ĐỊNH THƯ MỤC GỐC ────────────────────────
# Tìm thư mục chứa data.yaml (root của YOLO dataset)
yaml_candidates = glob.glob(f"{EXTRACT_DIR}/**/data.yaml", recursive=True)
if not yaml_candidates:
    raise FileNotFoundError("❌ Không tìm thấy data.yaml trong zip!")

DATASET_ROOT = str(Path(yaml_candidates[0]).parent)
print(f"📁 Dataset root: {DATASET_ROOT}")

# In cấu trúc để kiểm tra
print("\nCấu trúc thư mục:")
print("-" * 45)
for root, dirs, flist in os.walk(DATASET_ROOT):
    dirs[:] = sorted([d for d in dirs if not d.startswith(".")])
    level   = root.replace(DATASET_ROOT, "").count(os.sep)
    if level > 2: continue
    indent = "  " * level
    n = len([f for f in flist if not f.startswith(".")])
    print(f"{indent}{os.path.basename(root)}/  [{n} files]")
    for f in sorted(flist)[:3]:
        print(f"{indent}  {f}")
    if n > 3:
        print(f"{indent}  ... (+{n-3} files)")
print("-" * 45 + "\n")

# ── 5. ĐỌC VÀ XÁC NHẬN DỮ LIỆU ─────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def find_files(folder, exts=None):
    """Tìm tất cả file trong folder (recursive), lọc theo ext nếu có."""
    result = []
    for p in Path(folder).rglob("*"):
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                result.append(p)
    return sorted(result)

# Tìm tất cả ảnh và label
all_images = find_files(DATASET_ROOT, IMG_EXTS)
all_labels = [p for p in find_files(DATASET_ROOT, {".txt"})
              if p.name not in ("classes.txt", "notes.txt")]

print(f"📸 Tổng ảnh  : {len(all_images)}")
print(f"🏷️  Tổng label: {len(all_labels)}\n")

if len(all_images) == 0:
    raise RuntimeError("❌ Không tìm thấy ảnh! Kiểm tra lại file zip.")

# ── 6. GHÉP CẶP ẢNH ↔ LABEL ─────────────────────────────────
# Chiến lược: label ở thư mục "labels/..." tương ứng với "images/..."
# VD: images/train/Cars0.png → labels/train/Cars0.txt
# Build index: stem → label path

label_by_stem = {}
for lbl in all_labels:
    label_by_stem[lbl.stem] = lbl

paired    = []   # (img_path, lbl_path)
no_label  = []

for img in all_images:
    # Thử 1: cùng stem
    lbl = label_by_stem.get(img.stem)

    # Thử 2: thay "images" → "labels" trong path
    if lbl is None:
        candidate = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        if candidate.exists():
            lbl = candidate

    if lbl and lbl.exists():
        # Xác nhận format YOLO hợp lệ: "class xc yc w h"
        try:
            content = lbl.read_text().strip()
            if content:
                first = content.split("\n")[0].split()
                if len(first) == 5 and all(_is_num(x) for x in first):
                    paired.append((img, lbl))
                    continue
        except:
            pass
    no_label.append(img)

def _is_num(s):
    try: float(s); return True
    except: return False

# Gán lại sau khi định nghĩa _is_num
paired, no_label = [], []
for img in all_images:
    lbl = label_by_stem.get(img.stem)
    if lbl is None:
        candidate = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        if candidate.exists():
            lbl = candidate
    if lbl and lbl.exists():
        try:
            content = lbl.read_text().strip()
            if content:
                first = content.split("\n")[0].split()
                if len(first) == 5:
                    paired.append((img, lbl))
                    continue
        except:
            pass
    no_label.append(img)

print(f"✅ Cặp ảnh-label hợp lệ: {len(paired)}")
print(f"⚠️  Ảnh không có label  : {len(no_label)}")

if len(paired) == 0:
    # Debug chi tiết
    print("\n🔍 DEBUG — vài ảnh đầu:")
    for img in all_images[:3]:
        print(f"  img : {img}")
        lbl = label_by_stem.get(img.stem)
        print(f"  lbl : {lbl}")
    raise RuntimeError("❌ Không ghép được cặp ảnh-label. Xem DEBUG ở trên.")

# ── 7. PHÁT HIỆN SPLIT CÓ SẴN ───────────────────────────────
# Kiểm tra xem dataset đã chia train/val chưa bằng cách
# xem đường dẫn ảnh có chứa /train/ hay /val/ không

def detect_split(img_path: Path) -> str:
    parts = set(img_path.parts)
    if "train" in parts: return "train"
    if "val"   in parts: return "val"
    if "valid" in parts: return "val"
    if "test"  in parts: return "test"
    return "train"  # mặc định

split_counts = {}
for img, _ in paired:
    s = detect_split(img)
    split_counts[s] = split_counts.get(s, 0) + 1

print(f"\n📊 Phân bố split gốc: {split_counts}")
pre_split = len(split_counts) >= 2

# ── 8. HIỂN THỊ MẪU DỮ LIỆU ────────────────────────────────
print("\n🖼️  Hiển thị mẫu dữ liệu...")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Mẫu dữ liệu — License Plate Recognition", fontsize=14)

for ax, (img_p, lbl_p) in zip(axes.flat, paired[:6]):
    img     = cv2.imread(str(img_p))
    if img is None: ax.axis("off"); continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img.shape[:2]
    for line in lbl_p.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 5:
            _, xc, yc, bw, bh = map(float, parts)
            x1 = int((xc - bw/2)*w);  y1 = int((yc - bh/2)*h)
            x2 = int((xc + bw/2)*w);  y2 = int((yc + bh/2)*h)
            cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (255,60,60), 3)
    ax.imshow(img_rgb)
    ax.set_title(f"{img_p.name[:20]}  {w}×{h}", fontsize=9)
    ax.axis("off")

for ax in axes.flat: ax.axis("off")
plt.tight_layout()
plt.savefig("/content/data_samples.png", dpi=110, bbox_inches="tight")
plt.show()
print("✅ Đã hiển thị mẫu\n")

# ── 9. XÂY DỰNG YOLO DATASET CHUẨN ─────────────────────────
# Dataset archive__2_.zip có cấu trúc:
#   License-Plate-Data/train/images + labels  (346 ảnh)
#   License-Plate-Data/test/images  + labels  (87 ảnh) → dùng làm val
# Không có thư mục val/ riêng → map test/ → val/

YOLO_DIR = Path("/content/yolo_dataset")
for split in ["train", "val"]:
    (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

print("📂 Đang tổ chức YOLO dataset...")

# Map: train → train, test → val
SPLIT_MAP = {"train": "train", "test": "val", "valid": "val", "val": "val"}

for img_p, lbl_p in paired:
    # Xác định split gốc từ đường dẫn
    raw_split = next(
        (p for p in img_p.parts if p in SPLIT_MAP), None)
    dest_split = SPLIT_MAP.get(raw_split, "train")

    shutil.copy2(img_p, YOLO_DIR / dest_split / "images" / img_p.name)
    shutil.copy2(lbl_p, YOLO_DIR / dest_split / "labels" / lbl_p.name)

# Kiểm tra kết quả
print()
for split in ["train", "val"]:
    n_i = len(list((YOLO_DIR / split / "images").glob("*")))
    n_l = len(list((YOLO_DIR / split / "labels").glob("*.txt")))
    status = "✅" if n_i > 0 else "⚠️ "
    print(f"  {status} {split:5s}: {n_i:4d} ảnh | {n_l:4d} label")

n_train = len(list((YOLO_DIR / "train" / "images").glob("*")))
n_val   = len(list((YOLO_DIR / "val"   / "images").glob("*")))
if n_train == 0:
    raise RuntimeError("❌ Thư mục train rỗng! Kiểm tra cấu trúc zip.")
if n_val == 0:
    raise RuntimeError("❌ Thư mục val rỗng! Không tìm thấy test/ hoặc val/ trong zip.")

# ── 10. TẠO DATA.YAML ────────────────────────────────────────
# Ghi thủ công thay vì yaml.dump để đảm bảo đúng thứ tự key
# (yaml.dump sắp xếp alpha → Ultralytics không tìm thấy 'val')
yaml_out = YOLO_DIR / "data.yaml"
yaml_content = (
    f"path: {str(YOLO_DIR)}\n"
    "train: train/images\n"
    "val: val/images\n"
    "nc: 1\n"
    "names:\n"
    "  0: license_plate\n"
)
with open(yaml_out, "w") as f:
    f.write(yaml_content)

# cfg dict để tham chiếu ở các bước sau
cfg = {"path": str(YOLO_DIR), "train": "train/images",
       "val": "val/images", "nc": 1, "names": ["license_plate"]}

print(f"\n✅ data.yaml:\n{open(yaml_out).read()}")

# ── 11. TRAINING YOLOv8s ─────────────────────────────────────
print("\n" + "="*60)
print("  TRAINING YOLOv8s  (Small — 11.2M params)")
print("="*60 + "\n")

model = YOLO("yolov8s.pt")   # pretrained COCO

results_train = model.train(
    data          = str(yaml_out),
    epochs        = 60,
    imgsz         = 640,
    batch         = 16,        # T4 16GB — tăng lên 32 nếu còn bộ nhớ
    lr0           = 0.01,
    lrf           = 0.005,
    optimizer     = "AdamW",
    cos_lr        = True,
    warmup_epochs = 3,
    mosaic        = 1.0,
    mixup         = 0.1,
    flipud        = 0.1,
    fliplr        = 0.5,
    degrees       = 5.0,
    translate     = 0.1,
    scale         = 0.5,
    shear         = 2.0,
    perspective   = 0.0005,
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    patience      = 15,
    save          = True,
    project       = "/content/runs",
    name          = "lpr_yolov8s",
    exist_ok      = True,
    pretrained    = True,
    verbose       = True,
)

BEST_PT = "/content/runs/lpr_yolov8s/weights/best.pt"
print(f"\n✅ Training xong! Best model: {BEST_PT}\n")

# ── 12. ĐÁNH GIÁ MÔ HÌNH ────────────────────────────────────
print("=" * 60)
print("  ĐÁNH GIÁ MÔ HÌNH")
print("=" * 60 + "\n")

best_model = YOLO(BEST_PT)

# Dùng val nếu không có test
eval_split = "val"  # dataset này dùng val/ (map từ test/)
metrics = best_model.val(
    data    = str(yaml_out),
    split   = eval_split,
    imgsz   = 640,
    conf    = 0.25,
    iou     = 0.6,
    verbose = True,
)
print("\n📊 KẾT QUẢ:")
print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"  Precision    : {metrics.box.mp:.4f}")
print(f"  Recall       : {metrics.box.mr:.4f}")

# Hiển thị biểu đồ
for plot in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
    p = f"/content/runs/lpr_yolov8s/{plot}"
    if os.path.exists(p):
        img = mpimg.imread(p)
        plt.figure(figsize=(14, 6))
        plt.imshow(img); plt.axis("off")
        plt.title(plot.replace(".png","").replace("_"," ").title())
        plt.tight_layout(); plt.show()

# ── 13. EASYOCR SETUP ────────────────────────────────────────
print("\n⏳ Đang tải EasyOCR model...")
reader = easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
print("✅ EasyOCR sẵn sàng\n")

# ── 14. HÀM PIPELINE ─────────────────────────────────────────
# ── Bảng sửa ký tự nhầm lẫn phổ biến ────────────────────────
# Khi vị trí là SỐ mà OCR đọc ra chữ → sửa thành số
_CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}
# Khi vị trí là CHỮ mà OCR đọc ra số → sửa thành chữ
_DIGIT_TO_CHAR = {
    "0": "O",
    "1": "I",
    "8": "B",
    "6": "G",
    "5": "S",
}

def _fix_pos(ch: str, expect_digit: bool) -> str:
    if expect_digit:
        return _CHAR_TO_DIGIT.get(ch, ch)
    return _DIGIT_TO_CHAR.get(ch, ch)

def fix_vn_plate(raw: str) -> str:
    """
    Hậu xử lý chuẩn hóa biển số xe Việt Nam.
    Hỗ trợ: 51A-96141 (ô tô), 51AB-1234 (xe máy).
    """
    s = raw.upper()
    s = re.sub(r"[\s\.,\-]", "", s)

    # 2 số + 1 chữ + 5 số → 51A-961.41
    m = re.match(r"^(\d{2})([A-Z])(\d{5})$", s)
    if m:
        p1 = "".join(_fix_pos(c, True)  for c in m.group(1))
        p2 = _fix_pos(m.group(2), False)
        p3 = "".join(_fix_pos(c, True)  for c in m.group(3))
        return f"{p1}{p2}-{p3[:3]}.{p3[3:]}"

    # 2 số + 1 chữ + 4 số → 51A-9614
    m = re.match(r"^(\d{2})([A-Z])(\d{4})$", s)
    if m:
        p1 = "".join(_fix_pos(c, True)  for c in m.group(1))
        p2 = _fix_pos(m.group(2), False)
        p3 = "".join(_fix_pos(c, True)  for c in m.group(3))
        return f"{p1}{p2}-{p3}"

    # 2 số + 2 chữ + 4-5 số → xe máy 51AB-1234
    m = re.match(r"^(\d{2})([A-Z]{2})(\d{4,5})$", s)
    if m:
        p1 = "".join(_fix_pos(c, True)  for c in m.group(1))
        p3 = "".join(_fix_pos(c, True)  for c in m.group(3))
        return f"{p1}{m.group(2)}-{p3}"

    return re.sub(r"[^A-Z0-9\-\.]", "", raw.upper())

def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """Grayscale + scale lên 400px + CLAHE + Otsu binarization + sharpen."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Scale lên tối thiểu 400px (bản cũ chỉ 200px — không đủ cho OCR)
    if w < 400:
        scale = 400 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    # Otsu binarization — phân tách nền/chữ rõ hơn
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Sharpen nhẹ
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(binary, -1, kernel)

def preprocess_plate_adaptive(crop: np.ndarray) -> np.ndarray:
    """Chiến lược dự phòng: adaptive threshold — tốt hơn khi ánh sáng không đều."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 400:
        scale = 400 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blurred, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

def clean_text(text: str) -> str:
    """Chỉ giữ A-Z, 0-9, dấu gạch ngang."""
    return re.sub(r"[^A-Z0-9\-]", "", text.upper().strip())

def detect_and_read(image_path: str,
                    yolo_model,
                    ocr_reader,
                    conf: float = 0.4):
    """
    Pipeline đầy đủ (đã cải tiến):
      YOLO detect → crop → đa chiến lược tiền xử lý
      → EasyOCR (beamWidth=10) → fix_vn_plate post-processing
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, []

    img_rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    yolo_results = yolo_model.predict(img, conf=conf, verbose=False)
    ALLOW        = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

    plates = []
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score      = float(box.conf[0])

            m   = 6
            cx1 = max(0, x1-m);  cy1 = max(0, y1-m)
            cx2 = min(img.shape[1], x2+m)
            cy2 = min(img.shape[0], y2+m)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            raw_text = ""
            # Thử 4 chiến lược, dừng khi có kết quả >= 4 ký tự
            strategies = [
                lambda: preprocess_plate(crop),
                lambda: preprocess_plate_adaptive(crop),
                lambda: crop,
                lambda: cv2.bitwise_not(preprocess_plate(crop)),
            ]
            for get_img in strategies:
                ocr_r = ocr_reader.readtext(
                    get_img(), allowlist=ALLOW,
                    batch_size=1, detail=1, paragraph=False,
                    beamWidth=10,  # thử nhiều ứng viên ký tự hơn
                )
                candidate = "".join(
                    clean_text(r[1]) for r in ocr_r if r[2] > 0.3)
                if len(candidate) >= 4:
                    raw_text = candidate
                    break

            # Hậu xử lý chuẩn hóa biển số VN
            text = fix_vn_plate(raw_text) if raw_text else ""

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf_score,
                "text": text,
                "raw":  raw_text,
                "crop": crop,
            })

            # Vẽ kết quả lên ảnh
            color = (50, 220, 80)
            cv2.rectangle(img_rgb, (x1,y1), (x2,y2), color, 3)
            label = f"{text}  {conf_score:.2f}" if text else f"({conf_score:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            ty = max(y1-10, lh+4)
            cv2.rectangle(img_rgb, (x1, ty-lh-4), (x1+lw+4, ty+2), color, -1)
            cv2.putText(img_rgb, label, (x1+2, ty-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20,20,20), 2)

    return img_rgb, plates

# ── 15. DEMO KẾT QUẢ ────────────────────────────────────────
print("=" * 60)
print("  DEMO KẾT QUẢ — " + eval_split.upper())
print("=" * 60 + "\n")

demo_imgs = list((YOLO_DIR / eval_split / "images").glob("*"))[:9]
cols = 3
rows = max(1, (len(demo_imgs) + cols - 1) // cols)

fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
fig.suptitle("YOLOv8s Detection + EasyOCR Recognition", fontsize=15, y=1.01)
axes_flat = np.array(axes).flat

for ax, img_path in zip(axes_flat, demo_imgs):
    img_out, plates = detect_and_read(str(img_path), best_model, reader)
    if img_out is not None:
        ax.imshow(img_out)
        texts = ", ".join(p["text"] for p in plates if p["text"]) or "—"
        ax.set_title(f"Biển số: {texts}", fontsize=11)
    ax.axis("off")

for ax in list(axes_flat)[len(demo_imgs):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig("/content/detection_results.png", dpi=120, bbox_inches="tight")
plt.show()

# ── 16. ĐO TỐC ĐỘ INFERENCE ─────────────────────────────────
print("\n⏱️  Đo tốc độ inference...")
timed_imgs = demo_imgs[:min(20, len(demo_imgs))]
yolo_ms, ocr_ms = [], []

for img_path in timed_imgs:
    img = cv2.imread(str(img_path))
    t0  = time.time()
    res = best_model.predict(img, conf=0.4, verbose=False)
    yolo_ms.append((time.time()-t0)*1000)
    for r in res:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                t1 = time.time()
                reader.readtext(preprocess_plate(crop))
                ocr_ms.append((time.time()-t1)*1000)

avg_yolo  = np.mean(yolo_ms) if yolo_ms else 0
avg_ocr   = np.mean(ocr_ms)  if ocr_ms  else 0
avg_total = avg_yolo + avg_ocr
print(f"  YOLOv8s detection : {avg_yolo:.1f} ms/ảnh")
print(f"  EasyOCR per plate : {avg_ocr:.1f} ms/biển số")
print(f"  Tổng pipeline     : {avg_total:.1f} ms/ảnh")
print(f"  FPS ước tính      : {1000/max(avg_total,1):.1f} FPS")

# ── 17. THỬ NGHIỆM ẢNH UPLOAD ───────────────────────────────
print("\n📷 Upload ảnh xe của bạn để thử (hoặc bấm Cancel):")
try:
    uploaded_imgs = files.upload()
    for fname, fdata in uploaded_imgs.items():
        tmp = f"/tmp/{fname}"
        open(tmp, "wb").write(fdata)
        img_out, plates = detect_and_read(tmp, best_model, reader, conf=0.3)
        if img_out is None: continue
        plt.figure(figsize=(12,8))
        plt.imshow(img_out); plt.axis("off")
        if plates:
            title = "\n".join(
                f"#{i+1}: {p['text']}  (conf {p['conf']:.2f})"
                for i,p in enumerate(plates))
            for i,p in enumerate(plates):
                print(f"  #{i+1}: {p['text']:15s} | conf: {p['conf']:.3f}")
        else:
            title = "Không phát hiện biển số — thử giảm conf"
        plt.title(title, fontsize=12)
        plt.tight_layout(); plt.show()
except Exception:
    print("⏭️  Bỏ qua upload")

# ── 18. XUẤT MODEL ───────────────────────────────────────────
print("\n" + "="*60)
print("  XUẤT MODEL")
print("="*60)

# Export ONNX
best_model.export(format="onnx", imgsz=640, opset=12)
print("✅ Xuất ONNX xong")

# Đóng gói tất cả output
zip_out = "/content/license_plate_yolov8s_model.zip"
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(BEST_PT, "weights/best.pt")
    onnx_p = BEST_PT.replace(".pt", ".onnx")
    if os.path.exists(onnx_p):
        zf.write(onnx_p, "weights/best.onnx")
    zf.write("/content/detection_results.png", "detection_results.png")
    zf.write("/content/data_samples.png",      "data_samples.png")
    # Lưu lại data.yaml sạch
    zf.write(str(yaml_out), "data.yaml")

files.download(zip_out)

# ── 19. TỔNG KẾT ────────────────────────────────────────────
print("\n" + "="*60)
print("  ✅ HOÀN TẤT!")
print("="*60)
print(f"\n  Dataset   : {len(paired)} cặp ảnh-label")
print(f"  Train/Val : {split_counts}")
print(f"  Model     : {BEST_PT}")
print(f"  mAP@0.5   : {metrics.box.map50:.4f}")
print(f"  Precision : {metrics.box.mp:.4f}")
print(f"  Recall    : {metrics.box.mr:.4f}")
print(f"  FPS       : {1000/max(avg_total,1):.1f}")