# ============================================================
#  LICENSE PLATE RECOGNITION — Faster R-CNN + EasyOCR
#  Dataset: archive__2_.zip  (cùng dataset với YOLOv8s)
#  Backbone: ResNet-50 + FPN  (pretrained COCO)
#  Chạy trên Google Colab T4 GPU (~40-60 phút)
#
#  So sánh với: lpr_fixed.py (YOLOv8s)
#  Mục tiêu: đánh giá trade-off độ chính xác vs tốc độ
#  giữa two-stage (Faster R-CNN) và one-stage (YOLOv8s)
# ============================================================

# ── 0. CÀI ĐẶT ──────────────────────────────────────────────
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install",
                "torchvision", "easyocr", "pycocotools", "-q"], check=True)

import os, glob, shutil, re, time, json, zipfile
import cv2, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import easyocr
from google.colab import files

# ── 1. KIỂM TRA GPU ─────────────────────────────────────────
print("=" * 60)
print("  LICENSE PLATE RECOGNITION — Faster R-CNN + EasyOCR")
print("=" * 60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("\n⚠️  Không có GPU — Runtime > Change runtime type > T4 GPU")
print()

# ── 2. UPLOAD FILE ZIP ──────────────────────────────────────
print("📂 Upload file archive__2_.zip:")
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
yaml_candidates = glob.glob(f"{EXTRACT_DIR}/**/data.yaml", recursive=True)
if not yaml_candidates:
    raise FileNotFoundError("❌ Không tìm thấy data.yaml trong zip!")

DATASET_ROOT = str(Path(yaml_candidates[0]).parent)
print(f"📁 Dataset root: {DATASET_ROOT}\n")

# ── 5. ĐỌC DỮ LIỆU ──────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_files(folder, exts=None):
    result = []
    for p in Path(folder).rglob("*"):
        if p.is_file():
            if exts is None or p.suffix.lower() in exts:
                result.append(p)
    return sorted(result)

all_images = find_files(DATASET_ROOT, IMG_EXTS)
all_labels = [p for p in find_files(DATASET_ROOT, {".txt"})
              if p.name not in ("classes.txt", "notes.txt")]

print(f"📸 Tổng ảnh  : {len(all_images)}")
print(f"🏷️  Tổng label: {len(all_labels)}\n")

# ── 6. GHÉP CẶP ẢNH ↔ LABEL ─────────────────────────────────
def _is_num(s):
    try: float(s); return True
    except: return False

label_by_stem = {lbl.stem: lbl for lbl in all_labels}

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
print(f"⚠️  Ảnh không có label  : {len(no_label)}\n")

if len(paired) == 0:
    raise RuntimeError("❌ Không ghép được cặp ảnh-label!")

# ── 7. CHIA TRAIN / VAL ──────────────────────────────────────
# Dataset archive__2_.zip có train/ và test/ → map test → val
SPLIT_MAP = {"train": "train", "test": "val", "valid": "val", "val": "val"}

train_pairs, val_pairs = [], []
for img_p, lbl_p in paired:
    raw_split = next((p for p in img_p.parts if p in SPLIT_MAP), None)
    dest = SPLIT_MAP.get(raw_split, "train")
    if dest == "train":
        train_pairs.append((img_p, lbl_p))
    else:
        val_pairs.append((img_p, lbl_p))

print(f"✅ Train: {len(train_pairs)} ảnh | Val: {len(val_pairs)} ảnh\n")

# ── 8. HIỂN THỊ MẪU DỮ LIỆU ────────────────────────────────
print("🖼️  Hiển thị mẫu dữ liệu...")

def yolo_to_xyxy(box_yolo, img_w, img_h):
    """Chuyển YOLO format (cx cy w h norm) → pixel (x1 y1 x2 y2)."""
    cx, cy, bw, bh = box_yolo
    x1 = int((cx - bw/2) * img_w)
    y1 = int((cy - bh/2) * img_h)
    x2 = int((cx + bw/2) * img_w)
    y2 = int((cy + bh/2) * img_h)
    return x1, y1, x2, y2

samples = train_pairs[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Mẫu dữ liệu — Faster R-CNN Input", fontsize=14)
for ax, (img_p, lbl_p) in zip(axes.flat, samples):
    img = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    ax.imshow(img)
    for line in lbl_p.read_text().strip().split("\n"):
        parts = line.split()
        if len(parts) == 5:
            _, cx, cy, bw, bh = map(float, parts)
            x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, bw, bh), w, h)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor="#38b6ff",
                                       facecolor="none")
            ax.add_patch(rect)
    ax.set_title(img_p.name[:25], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig("/content/sample_data.png", dpi=100, bbox_inches="tight")
plt.show()
print("✅ Lưu ảnh mẫu: /content/sample_data.png\n")

# ── 9. CUSTOM DATASET CLASS ──────────────────────────────────
class LicensePlateDataset(Dataset):
    """
    Dataset cho Faster R-CNN.
    Faster R-CNN yêu cầu target là dict:
      {"boxes": Tensor[N,4] (x1y1x2y2), "labels": Tensor[N] (int)}
    Khác với YOLO nhận file .txt riêng.
    """
    def __init__(self, pairs, transforms=None):
        self.pairs      = pairs
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, lbl_p = self.pairs[idx]

        # Đọc ảnh → PIL → Tensor
        img = Image.open(str(img_p)).convert("RGB")
        w, h = img.size

        # Đọc YOLO labels → chuyển sang Pascal VOC (x1 y1 x2 y2) pixel
        boxes, labels = [], []
        for line in lbl_p.read_text().strip().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, bw, bh), w, h)
            # Clamp để tránh bbox ra ngoài ảnh
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # 1 = license_plate (0 = background)

        if not boxes:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
        }

        img_tensor = F.to_tensor(img)   # [3, H, W], float [0,1]

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target


def collate_fn(batch):
    """Custom collate vì mỗi ảnh có số bbox khác nhau."""
    return tuple(zip(*batch))


# ── 10. XÂY DỰNG MÔ HÌNH FASTER R-CNN ──────────────────────
print("=" * 60)
print("  XÂY DỰNG MÔ HÌNH FASTER R-CNN")
print("=" * 60)

def build_model(num_classes: int = 2):
    """
    Faster R-CNN với backbone ResNet-50 + FPN.
    Pretrained trên COCO (91 classes) → thay head → fine-tune.

    Kiến trúc:
      ResNet-50 → FPN (neck) → RPN → RoI Align → Classifier Head
    num_classes = 2: 0=background, 1=license_plate
    """
    # Load pretrained weights COCO
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    # Thay thế classifier head (Box Predictor)
    # in_features: số features từ RoI Pooling = 1024
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

model = build_model(num_classes=2)
model.to(device)

n_params = sum(p.numel() for p in model.parameters())
n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n✅ Mô hình: Faster R-CNN (ResNet-50 + FPN)")
print(f"   Tổng parameters  : {n_params:,}")
print(f"   Trainable params : {n_train:,}")
print(f"   Device           : {device}\n")

# ── 11. DATALOADER ───────────────────────────────────────────
BATCH_SIZE = 4    # giảm xuống 2 nếu VRAM < 8GB

train_ds = LicensePlateDataset(train_pairs)
val_ds   = LicensePlateDataset(val_pairs)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=2, pin_memory=True)

print(f"✅ DataLoader: {len(train_loader)} train batches | {len(val_loader)} val batches\n")

# ── 12. OPTIMIZER & SCHEDULER ───────────────────────────────
# SGD với momentum — chuẩn cho Faster R-CNN
optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005, momentum=0.9, weight_decay=0.0005
)
# StepLR: giảm lr 10x sau mỗi 10 epoch
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1
)

# ── 13. TRAINING ─────────────────────────────────────────────
print("=" * 60)
print("  BẮT ĐẦU TRAINING FASTER R-CNN")
print("=" * 60)

NUM_EPOCHS  = 25   # đủ để converge với dataset nhỏ ~346 ảnh
SAVE_DIR    = Path("/content/fasterrcnn_output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──
    model.train()
    train_losses = []
    t0 = time.time()

    for imgs, targets in train_loader:
        imgs    = [img.to(device)   for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        # Faster R-CNN trả về dict losses:
        # loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_losses.append(loss.item())

    lr_scheduler.step()

    # ── Validation Loss ──
    model.train()   # Faster R-CNN chỉ tính loss ở train mode
    val_losses = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs    = [img.to(device)   for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            val_losses.append(sum(loss_dict.values()).item())

    avg_train = np.mean(train_losses)
    avg_val   = np.mean(val_losses)
    elapsed   = time.time() - t0

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)

    print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] "
          f"train={avg_train:.4f}  val={avg_val:.4f}  "
          f"lr={optimizer.param_groups[0]['lr']:.5f}  "
          f"time={elapsed:.1f}s")

    # Lưu best model
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), SAVE_DIR / "best.pth")
        print(f"  💾 Lưu best model (val_loss={avg_val:.4f})")

# Lưu model cuối cùng
torch.save(model.state_dict(), SAVE_DIR / "last.pth")
print(f"\n✅ Training xong! Best val_loss: {best_val_loss:.4f}")

# ── 14. VẼ LOSS CURVE ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history["train_loss"], label="Train Loss", color="#38b6ff", linewidth=2)
ax.plot(history["val_loss"],   label="Val Loss",   color="#ff6b6b", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Faster R-CNN — Training Loss Curve")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_DIR / "loss_curve.png", dpi=120, bbox_inches="tight")
plt.show()
print("✅ Lưu loss curve:", SAVE_DIR / "loss_curve.png")

# ── 15. ĐÁNH GIÁ — IoU, Precision, Recall, mAP ─────────────
print("\n" + "=" * 60)
print("  ĐÁNH GIÁ MÔ HÌNH")
print("=" * 60)

def compute_iou(box_a, box_b):
    """Tính IoU giữa 2 bbox [x1,y1,x2,y2]."""
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def evaluate_model(model, loader, iou_thresh=0.5, score_thresh=0.5):
    """
    Tính Precision, Recall, F1, mAP@50 trên tập val.
    Logic:
      - Với mỗi ảnh, lấy predictions có confidence > score_thresh
      - Match với ground truth theo IoU > iou_thresh
      - TP = matched, FP = dư prediction, FN = thiếu GT
    """
    model.eval()
    all_tp = all_fp = all_fn = 0
    all_precisions = []
    inference_times = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]

            t0  = time.time()
            preds = model(imgs)
            inference_times.append((time.time() - t0) / len(imgs) * 1000)

            for pred, target in zip(preds, targets):
                # Filter predictions theo score
                keep   = pred["scores"] > score_thresh
                p_boxes = pred["boxes"][keep].cpu().numpy()
                g_boxes = target["boxes"].cpu().numpy()

                tp = fp = 0
                matched_gt = set()

                for pb in p_boxes:
                    best_iou, best_j = 0, -1
                    for j, gb in enumerate(g_boxes):
                        if j in matched_gt:
                            continue
                        iou = compute_iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_j = iou, j
                    if best_iou >= iou_thresh:
                        tp += 1; matched_gt.add(best_j)
                    else:
                        fp += 1

                fn = len(g_boxes) - len(matched_gt)
                all_tp += tp; all_fp += fp; all_fn += fn

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                all_precisions.append(prec)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    map50     = float(np.mean(all_precisions))
    avg_ms    = float(np.mean(inference_times))

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "mAP@50":    map50,
        "avg_ms":    avg_ms,
    }

# Load best model để evaluate
model.load_state_dict(torch.load(SAVE_DIR / "best.pth", map_location=device))
metrics = evaluate_model(model, val_loader)

print(f"\n📊 Kết quả đánh giá (Val set — IoU@0.5):")
print(f"   Precision  : {metrics['precision']:.4f}")
print(f"   Recall     : {metrics['recall']:.4f}")
print(f"   F1-Score   : {metrics['f1']:.4f}")
print(f"   mAP@50     : {metrics['mAP@50']:.4f}")
print(f"   Inference  : {metrics['avg_ms']:.1f} ms/ảnh")

# Lưu metrics ra JSON (dùng để so sánh với YOLOv8s)
with open(SAVE_DIR / "metrics.json", "w") as f:
    json.dump({**metrics, "model": "Faster R-CNN ResNet-50 FPN",
               "epochs": NUM_EPOCHS, "train_size": len(train_pairs),
               "val_size": len(val_pairs)}, f, indent=2)
print(f"\n✅ Lưu metrics: {SAVE_DIR}/metrics.json")

# ── 16. EASYOCR SETUP ────────────────────────────────────────
print("\n⏳ Khởi tạo EasyOCR (lần đầu ~1 phút do download model)...")
ocr_reader = easyocr.Reader(["en"], gpu=(device.type == "cuda"),
                             verbose=False, recog_network="english_g2")
print("✅ EasyOCR sẵn sàng\n")

# ── 17. HÀM POST-PROCESSING BIỂN SỐ VN ─────────────────────
_CHAR_TO_DIGIT = {"O":"0","Q":"0","D":"0","I":"1","L":"1",
                  "Z":"2","S":"5","G":"6","B":"8"}
_DIGIT_TO_CHAR = {"0":"O","1":"I","8":"B","6":"G","5":"S"}

def _fix_pos(ch, expect_digit):
    if expect_digit: return _CHAR_TO_DIGIT.get(ch, ch)
    return _DIGIT_TO_CHAR.get(ch, ch)

def fix_vn_plate(raw):
    s = raw.upper()
    s = re.sub(r"[\s\.,\-]", "", s)
    m = re.match(r"^(\d{2})([A-Z])(\d{5})$", s)
    if m:
        p1 = "".join(_fix_pos(c, True)  for c in m.group(1))
        p2 = _fix_pos(m.group(2), False)
        p3 = "".join(_fix_pos(c, True)  for c in m.group(3))
        return f"{p1}{p2}-{p3[:3]}.{p3[3:]}"
    m = re.match(r"^(\d{2})([A-Z])(\d{4})$", s)
    if m:
        p1 = "".join(_fix_pos(c, True)  for c in m.group(1))
        p2 = _fix_pos(m.group(2), False)
        p3 = "".join(_fix_pos(c, True)  for c in m.group(3))
        return f"{p1}{p2}-{p3}"
    m = re.match(r"^(\d{2})([A-Z]{2})(\d{4,5})$", s)
    if m:
        p1 = "".join(_fix_pos(c, True)  for c in m.group(1))
        p3 = "".join(_fix_pos(c, True)  for c in m.group(3))
        return f"{p1}{m.group(2)}-{p3}"
    return re.sub(r"[^A-Z0-9\-\.]", "", raw.upper())

def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 400:
        scale = 400 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    return cv2.filter2D(binary, -1, kernel)

def preprocess_adaptive(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 400:
        scale = 400 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

def ocr_plate(crop):
    ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    raw_text = ""
    for get_img in [lambda: preprocess_plate(crop),
                    lambda: preprocess_adaptive(crop),
                    lambda: crop,
                    lambda: cv2.bitwise_not(preprocess_plate(crop))]:
        items = ocr_reader.readtext(
            get_img(), allowlist=ALLOW,
            batch_size=1, detail=1, paragraph=False, beamWidth=10)
        candidate = "".join(
            re.sub(r"[^A-Z0-9\-]", "", r[1].upper())
            for r in items if r[2] > 0.3)
        if len(candidate) >= 4:
            raw_text = candidate
            break
    return fix_vn_plate(raw_text) if raw_text else "", raw_text

# ── 18. HÀM DETECT + OCR (PIPELINE ĐẦY ĐỦ) ─────────────────
def detect_and_read(image_path, score_thresh=0.5):
    """
    Pipeline Faster R-CNN:
      Ảnh → ResNet-50 → FPN → RPN proposals → RoI Align
      → Box Predictor → NMS → crop → EasyOCR → fix_vn_plate
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None, []

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil  = Image.fromarray(img_rgb)
    img_t    = F.to_tensor(img_pil).to(device)

    model.eval()
    with torch.no_grad():
        preds = model([img_t])

    pred   = preds[0]
    keep   = pred["scores"] > score_thresh
    boxes  = pred["boxes"][keep].cpu().numpy().astype(int)
    scores = pred["scores"][keep].cpu().numpy()

    plates = []
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        m   = 6
        cx1 = max(0, x1-m); cy1 = max(0, y1-m)
        cx2 = min(img_bgr.shape[1], x2+m)
        cy2 = min(img_bgr.shape[0], y2+m)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        text, raw = ocr_plate(crop)
        plates.append({"bbox": (x1,y1,x2,y2), "conf": float(score),
                        "text": text, "raw": raw, "crop": crop})

        color = (50, 182, 255)
        cv2.rectangle(img_rgb, (x1,y1), (x2,y2), color, 3)
        label = f"{text}  {score:.2f}" if text else f"({score:.2f})"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .85, 2)
        ty = max(y1-10, lh+4)
        cv2.rectangle(img_rgb, (x1,ty-lh-4),(x1+lw+4,ty+2), color, -1)
        cv2.putText(img_rgb, label, (x1+2, ty-2),
                    cv2.FONT_HERSHEY_SIMPLEX, .85, (20,20,20), 2)

    return img_rgb, plates

# ── 19. DEMO KẾT QUẢ ────────────────────────────────────────
print("=" * 60)
print("  DEMO KẾT QUẢ — VAL SET")
print("=" * 60)

demo_imgs = [p for p, _ in val_pairs[:9]]
n = len(demo_imgs)
cols = 3; rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
fig.suptitle("Faster R-CNN — Kết quả nhận diện biển số", fontsize=14)

for i, ax in enumerate(axes.flat):
    if i < n:
        img_out, plates = detect_and_read(demo_imgs[i])
        if img_out is not None:
            ax.imshow(img_out)
            title = " | ".join(p["text"] for p in plates) or "Không phát hiện"
            ax.set_title(title, fontsize=9, color="#38b6ff")
    ax.axis("off")

plt.tight_layout()
plt.savefig(SAVE_DIR / "demo_results.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"✅ Lưu demo: {SAVE_DIR}/demo_results.png\n")

# ── 20. ĐO TỐC ĐỘ INFERENCE ─────────────────────────────────
print("=" * 60)
print("  ĐO TỐC ĐỘ INFERENCE")
print("=" * 60)

test_imgs  = [p for p, _ in val_pairs[:20]]
n_warmup   = 3
times_ms   = []

model.eval()
with torch.no_grad():
    for i, img_p in enumerate(test_imgs):
        img_t = F.to_tensor(Image.open(str(img_p)).convert("RGB")).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model([img_t])
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            times_ms.append(elapsed)

avg_ms = np.mean(times_ms)
std_ms = np.std(times_ms)
fps    = 1000 / avg_ms

print(f"\n🚀 Tốc độ inference (GPU: {device}):")
print(f"   Trung bình : {avg_ms:.1f} ms/ảnh")
print(f"   Std dev    : {std_ms:.1f} ms")
print(f"   FPS        : {fps:.1f}")

# Cập nhật metrics với FPS thực đo
with open(SAVE_DIR / "metrics.json") as f:
    saved = json.load(f)
saved.update({"fps": round(fps, 2), "avg_inference_ms": round(avg_ms, 2)})
with open(SAVE_DIR / "metrics.json", "w") as f:
    json.dump(saved, f, indent=2)

# ── 21. THỬ NGHIỆM ẢNH UPLOAD ───────────────────────────────
print("\n" + "=" * 60)
print("  THỬ NGHIỆM VỚI ẢNH CỦA BẠN")
print("=" * 60)

print("📂 Upload ảnh muốn thử (jpg/png):")
try:
    uploaded_test = files.upload()
    for fname in uploaded_test:
        with open(fname, "wb") as f:
            f.write(uploaded_test[fname])
        img_out, plates = detect_and_read(fname)
        if img_out is not None:
            plt.figure(figsize=(12, 7))
            plt.imshow(img_out)
            plt.axis("off")
            title = " | ".join(p["text"] for p in plates) or "Không phát hiện biển số"
            plt.title(f"Faster R-CNN: {title}", fontsize=13, color="#38b6ff")
            plt.tight_layout()
            plt.savefig(SAVE_DIR / f"test_{fname}", dpi=120, bbox_inches="tight")
            plt.show()
            for p in plates:
                print(f"  🚗 {p['text']}  (conf: {p['conf']:.3f})")
                if p["raw"] and p["raw"] != p["text"]:
                    print(f"     OCR thô: {p['raw']}")
except Exception as e:
    print(f"⚠️  Bỏ qua upload test: {e}")

# ── 22. XUẤT MODEL ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  XUẤT MODEL")
print("=" * 60)

# Đóng gói để download
output_zip = "/content/fasterrcnn_model.zip"
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in SAVE_DIR.glob("*"):
        zf.write(str(f), f.name)

print(f"✅ Đóng gói xong: {output_zip}")
print("📥 Đang tải xuống...")
files.download(output_zip)

# ── 23. TỔNG KẾT ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TỔNG KẾT — FASTER R-CNN")
print("=" * 60)
print(f"  Mô hình     : Faster R-CNN (ResNet-50 + FPN)")
print(f"  Backbone    : ResNet-50 pretrained COCO")
print(f"  Parameters  : {n_params:,}")
print(f"  Epochs      : {NUM_EPOCHS}")
print(f"  Train/Val   : {len(train_pairs)}/{len(val_pairs)}")
print(f"  Precision   : {metrics['precision']:.4f}")
print(f"  Recall      : {metrics['recall']:.4f}")
print(f"  F1-Score    : {metrics['f1']:.4f}")
print(f"  mAP@50      : {metrics['mAP@50']:.4f}")
print(f"  FPS         : {fps:.1f}")
print(f"  Best weights: {SAVE_DIR}/best.pth")
print("=" * 60)
