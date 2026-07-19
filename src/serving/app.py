"""
FastAPI inference server for the Distill-LightOVD student model.
"""

import io

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from PIL import Image

from src.student.detector import StudentDetector
from src.evaluation.decode import decode_predictions
from src.data.dataset import CLASS_NAMES

app = FastAPI(title="Distill-LightOVD Inference API")

DEVICE = "cpu"  # edge-target inference is CPU by design
CHECKPOINT_PATH = "checkpoints/latest.pt"

model = None
transform = T.Compose(
    [
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.on_event("startup")
def load_model():
    """Load the trained student model once, at server startup."""
    global model
    model = StudentDetector(num_classes=len(CLASS_NAMES)).to(DEVICE)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    except FileNotFoundError:
        print(
            f"Warning: no checkpoint found at {CHECKPOINT_PATH}. Using untrained weights."
        )
    model.eval()


@app.get("/health")
def health_check():
    """Basic liveness check."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf_threshold: float = 0.2,
    nms_threshold: float = 0.3,
):
    """
    Run object detection on an uploaded image.

    Returns a list of detections with box coordinates (in original image
    pixels), class labels, and confidence scores.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls_logits, reg_preds, centerness_preds = model(input_tensor)

    boxes, scores, classes = decode_predictions(
        cls_logits,
        reg_preds,
        centerness_preds,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )

    scale_x = orig_w / 320
    scale_y = orig_h / 320

    detections = []
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            {
                "label": CLASS_NAMES[cls_id.item()],
                "confidence": round(score.item(), 3),
                "box": [
                    round(x1 * scale_x, 1),
                    round(y1 * scale_y, 1),
                    round(x2 * scale_x, 1),
                    round(y2 * scale_y, 1),
                ],
            }
        )

    return JSONResponse({"filename": file.filename, "detections": detections})


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Distill-LightOVD</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --border: #30363d;
    --accent: #7ee787;
    --text: #e6edf3;
    --muted: #8b949e;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, sans-serif;
    margin: 0;
    padding: 40px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  h1 {
    font-size: 28px;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
  }
  .subtitle {
    color: var(--muted);
    margin-bottom: 32px;
    font-size: 14px;
  }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    width: 100%;
    max-width: 720px;
  }
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s;
  }
  .dropzone:hover { border-color: var(--accent); }
  input[type=file] { display: none; }
  button {
    background: var(--accent);
    color: #0d1117;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    margin-top: 16px;
    font-size: 14px;
  }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  #canvasWrap {
    margin-top: 24px;
    text-align: center;
  }
  canvas {
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .status { color: var(--muted); font-size: 13px; margin-top: 12px; }
  .detections {
    margin-top: 16px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .tag {
    background: #21262d;
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    color: var(--accent);
  }
</style>
</head>
<body>
  <h1>Distill-LightOVD</h1>
  <div class="subtitle">Lightweight edge object detector — live inference demo</div>

  <div class="panel">
    <div class="dropzone" id="dropzone">
      <div>Click to upload an image</div>
      <input type="file" id="fileInput" accept="image/*">
    </div>
    <button id="detectBtn" disabled>Run Detection</button>
    <div class="status" id="status"></div>
    <div id="canvasWrap"><canvas id="canvas"></canvas></div>
    <div class="detections" id="detections"></div>
  </div>

<script>
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const detectionsEl = document.getElementById('detections');
let currentFile = null;
let img = new Image();

dropzone.onclick = () => fileInput.click();

fileInput.onchange = () => {
  currentFile = fileInput.files[0];
  if (!currentFile) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(currentFile);
  detectBtn.disabled = false;
  statusEl.textContent = '';
  detectionsEl.innerHTML = '';
};

detectBtn.onclick = async () => {
  if (!currentFile) return;
  statusEl.textContent = 'Running inference...';
  detectBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res = await fetch('/detect?conf_threshold=0.2&nms_threshold=0.3', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();

    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 3;
    ctx.font = '16px Segoe UI';

    detectionsEl.innerHTML = '';
    data.detections.forEach(det => {
      const [x1, y1, x2, y2] = det.box;
      ctx.strokeStyle = '#7ee787';
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = '#7ee787';
      ctx.fillText(`${det.label} ${det.confidence}`, x1, y1 - 6);

      const tag = document.createElement('div');
      tag.className = 'tag';
      tag.textContent = `${det.label} (${det.confidence})`;
      detectionsEl.appendChild(tag);
    });

    statusEl.textContent = `${data.detections.length} object(s) detected`;
  } catch (err) {
    statusEl.textContent = 'Error running detection';
  }
  detectBtn.disabled = false;
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the visual detection dashboard."""
    return DASHBOARD_HTML
