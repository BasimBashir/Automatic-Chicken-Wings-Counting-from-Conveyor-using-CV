# Chicken Wings Counting System

An end-to-end computer vision system for detecting and counting chicken wings using **YOLOv8s** object detection. Supports image detection, video processing with line-crossing counting, and RTSP live stream monitoring through a web-based dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Training](#training)
- [Inference](#inference)
- [Web Application](#web-application)
- [Deployment](#deployment)
  - [Docker Hub — pre-built images](#docker-hub--pre-built-images)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [License](#license)

---

## Overview

This system provides three operational modes:

| Mode | Description |
|------|-------------|
| **Image** | Upload a single image, detect all chicken wings, return annotated result with count |
| **Video** | Upload a video file, track wings across frames, count each wing exactly once as it crosses a configurable ROI (Region of Interest) line |
| **Stream** | Connect to an RTSP camera feed for real-time wing detection and counting |

The detection model is **YOLOv8s** (small variant) trained on a single-class dataset (`Wings-Counting.`) sourced from [Roboflow](https://universe.roboflow.com/mydis-gqble/wings-counting/dataset/2). Tracking uses a centroid-based tracker that assigns persistent IDs to detected objects across frames.

---

## Project Structure

```
Chickens_Wings_Counting/
├── dataset/                        # Roboflow dataset (YOLO format)
│   ├── data.yaml                   # Dataset config (classes, paths)
│   ├── train/images/               # Training images
│   ├── train/labels/               # Training labels
│   ├── valid/images/               # Validation images
│   ├── valid/labels/               # Validation labels
│   ├── test/images/                # Test images
│   └── test/labels/                # Test labels
├── app/                            # FastAPI web application
│   ├── main.py                     # App entrypoint
│   ├── config.py                   # Settings (env-based)
│   ├── core/
│   │   ├── detector.py             # YOLOv8 model loading & inference
│   │   ├── tracker.py              # Centroid-based object tracker
│   │   ├── counter.py              # ROI line-crossing counter
│   │   ├── annotator.py            # Frame annotation (bboxes, trails, dashboard)
│   │   └── video_processor.py      # Background video/stream processor
│   ├── routers/
│   │   ├── image.py                # POST /api/image/detect
│   │   ├── video.py                # Video upload, playback, counting
│   │   ├── stream.py               # RTSP stream management
│   │   └── config_router.py        # Runtime config (ROI, confidence)
│   └── static/                     # Frontend (HTML/CSS/JS)
│       ├── index.html              # Image detection page
│       ├── video.html              # Video processing page
│       ├── stream.html             # Live stream page
│       ├── css/style.css
│       └── js/
├── train.py                        # Model training script
├── detect_and_count.py             # Standalone CLI for image/video inference
├── best.pt                         # Trained model weights (after training)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image
├── docker-compose.yml              # Docker Compose with GPU support
├── start.bat                       # Windows quick-start (Docker)
└── .env                            # Environment variables
```

---

## Prerequisites

- **Python** 3.10+
- **CUDA-compatible GPU** (recommended for training; CPU works for inference)
- **FFmpeg** (required for video re-encoding/download feature)
- **Docker** (optional, for containerized deployment)

### Hardware Recommendations

| Task | Minimum | Recommended |
|------|---------|-------------|
| Training | 8 GB VRAM GPU | 12+ GB VRAM GPU (RTX 3060+) |
| Inference | CPU (slow) | Any CUDA GPU |
| Web App | 4 GB RAM | 8+ GB RAM |

---

## Setup

### 1. Clone or download the project

```bash
cd Chickens_Wings_Counting
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `ultralytics` (YOLOv8), `fastapi`, `uvicorn`, `opencv-python-headless`, `numpy`, and other required packages.

### 4. Verify GPU availability (optional)

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## Training

The training script (`train.py`) trains YOLOv8s on the Roboflow dataset with optimized hyperparameters.

### Dataset

The dataset follows the standard Roboflow/YOLO format with `data.yaml` pointing to train/val/test splits:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['Wings-Counting.']
```

### Run training

```bash
python train.py
```

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `yolov8s.pt` | YOLOv8 Small, pretrained on COCO |
| Image Size | 512x512 | Input resolution |
| Epochs | 50 | Max training epochs |
| Early Stopping | 5 | Stop if no improvement for 5 epochs |
| Batch Size | 16 | Adjust down if GPU OOM |
| Optimizer | SGD | lr=0.01, momentum=0.937 |
| LR Scheduler | Cosine | With 3-epoch warmup |
| AMP | Enabled | Mixed precision for speed |

**Data Augmentation:**
- Mosaic (1.0), Mixup (0.1), Copy-Paste (0.1)
- HSV: hue=0.015, saturation=0.7, value=0.4
- Rotation (10 deg), Scale (0.5), Horizontal Flip (0.5)
- Mosaic disabled for last 10 epochs (`close_mosaic=10`)

**Loss Weights:**
- Box: 7.5 (bounding box regression)
- Cls: 0.5 (classification, lower since single class)
- DFL: 1.5 (distribution focal loss)

### Training Output

```
runs/detect/chicken_wings/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Per-epoch metrics
├── confusion_matrix.png
├── results.png          # Loss & mAP curves
├── val_batch*_pred.png  # Validation predictions
└── ...
```

### After training

Copy the best model to the project root:

```bash
# Windows
copy runs\detect\chicken_wings\weights\best.pt best.pt

# Linux/macOS
cp runs/detect/chicken_wings/weights/best.pt best.pt
```

### Adjusting for your hardware

If you run out of GPU memory, reduce `batch` size in `train.py`:

```python
batch=8,   # or batch=4 for very limited VRAM
```

To train on CPU (slow, not recommended):

```python
device="cpu",
```

---

## Inference

### CLI - Standalone Script

The `detect_and_count.py` script works without the web server for quick testing.

#### Detect wings in an image

```bash
python detect_and_count.py path/to/image.jpg
```

With options:

```bash
python detect_and_count.py path/to/image.jpg --conf 0.3 --model best.pt --save output.jpg
```

#### Count wings in a video

```bash
python detect_and_count.py path/to/video.mp4
```

With options:

```bash
python detect_and_count.py path/to/video.mp4 \
    --conf 0.25 \
    --roi 0.7 \
    --max-distance 40 \
    --max-disappeared 50 \
    --save annotated_output.mp4
```

#### Process an RTSP stream

```bash
python detect_and_count.py rtsp://user:pass@camera-ip:554/stream
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Path to image, video, or RTSP URL |
| `--model` | `best.pt` | Path to YOLOv8 model weights |
| `--conf` | `0.25` | Detection confidence threshold |
| `--roi` | `0.7` | ROI line position (0.0=top, 1.0=bottom) |
| `--max-distance` | `40` | Max pixel distance for tracker matching |
| `--max-disappeared` | `50` | Frames before dropping a lost track |
| `--save` | `None` | Output path for annotated result |

### Controls (during video/stream playback)

- Press **`q`** to quit

---

## Web Application

The web app provides a browser-based UI with three pages: Image, Video, and Stream.

### Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5580
```

Then open **http://localhost:5580** in your browser.

### Image Detection Page (`/`)

1. Drag-and-drop or click to upload an image
2. The server runs YOLOv8 detection and returns an annotated image
3. Displays wing count and side-by-side comparison (original vs annotated)
4. Download the annotated image

### Video Processing Page (`/video.html`)

1. Upload a video file (MP4, AVI, MOV, MKV)
2. Click **Play** to start processing
3. Click **Start Counting** to enable the ROI line-crossing counter
4. Watch the live MJPEG feed with bounding boxes, trails, and dashboard overlay
5. When complete, download the annotated output video (re-encoded to H.264)

### Live Stream Page (`/stream.html`)

1. Enter an RTSP URL (or configure via `.env`)
2. Click **Connect** to start the stream
3. Click **Start Counting** to enable counting
4. Adjust ROI position and confidence threshold in real time via sliders
5. Live stats: wing count, FPS

---

## Deployment

### Option 1: Direct (bare metal / VM)

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure best.pt is in the project root
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 5580
```

For production, use multiple workers (CPU-bound inference won't benefit much):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5580 --workers 1
```

### Option 2: Docker

#### Build and run locally

```bash
docker compose up --build
```

This builds the image (tagged as `basim123/chickens-wings-counter:latest`), starts the container on port **5580** with GPU passthrough (NVIDIA Container Toolkit required).

#### Quick start (Windows)

Double-click `start.bat` — it builds the container, waits for the server, and opens the browser automatically.

#### Docker Compose configuration

The `docker-compose.yml` mounts upload/output directories as volumes and passes environment variables:

```yaml
services:
  wing-counter:
    image: basim123/chickens-wings-counter:latest
    build: .
    ports:
      - "5580:5580"
    volumes:
      - ./app/uploads:/app/app/uploads
      - ./app/outputs:/app/app/outputs
    environment:
      - MODEL_PATH=best.pt
      - ROI_POSITION=0.7
      - CONFIDENCE=0.25
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

### Docker Hub — pre-built images

Pre-built images are published on Docker Hub under **`basim123/chickens-wings-counter`**:

| Tag | Base | Use when |
|-----|------|----------|
| `latest` | NVIDIA CUDA 12.6.2 | You have an NVIDIA GPU + NVIDIA Container Toolkit |
| `cpu` | Python 3.12 slim | No GPU / any machine |

---

#### Build and push (owner only)

```bash
# GPU image (default)
docker build -t basim123/chickens-wings-counter:latest .
docker push basim123/chickens-wings-counter:latest

# CPU image
docker build -f Dockerfile.cpu -t basim123/chickens-wings-counter:cpu .
docker push basim123/chickens-wings-counter:cpu
```

---

#### Pull and run — GPU

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker pull basim123/chickens-wings-counter:latest
```

Create a local folder for uploads/outputs, then run:

```bash
# Linux / macOS
docker run -d \
  --gpus all \
  --name wing-counter \
  -p 5580:5580 \
  -v "$(pwd)/uploads:/app/app/uploads" \
  -v "$(pwd)/outputs:/app/app/outputs" \
  -e MODEL_PATH=best.pt \
  -e ROI_POSITION=0.7 \
  -e CONFIDENCE=0.25 \
  --restart unless-stopped \
  basim123/chickens-wings-counter:latest
```

```powershell
# Windows PowerShell
docker run -d `
  --gpus all `
  --name wing-counter `
  -p 5580:5580 `
  -v "${PWD}/uploads:/app/app/uploads" `
  -v "${PWD}/outputs:/app/app/outputs" `
  -e MODEL_PATH=best.pt `
  -e ROI_POSITION=0.7 `
  -e CONFIDENCE=0.25 `
  --restart unless-stopped `
  basim123/chickens-wings-counter:latest
```

Open **http://localhost:5580** in your browser.

---

#### Pull and run — CPU

No GPU or NVIDIA toolkit needed.

```bash
docker pull basim123/chickens-wings-counter:cpu
```

```bash
# Linux / macOS
docker run -d \
  --name wing-counter \
  -p 5580:5580 \
  -v "$(pwd)/uploads:/app/app/uploads" \
  -v "$(pwd)/outputs:/app/app/outputs" \
  -e MODEL_PATH=best.pt \
  -e ROI_POSITION=0.7 \
  -e CONFIDENCE=0.25 \
  --restart unless-stopped \
  basim123/chickens-wings-counter:cpu
```

```powershell
# Windows PowerShell
docker run -d `
  --name wing-counter `
  -p 5580:5580 `
  -v "${PWD}/uploads:/app/app/uploads" `
  -v "${PWD}/outputs:/app/app/outputs" `
  -e MODEL_PATH=best.pt `
  -e ROI_POSITION=0.7 `
  -e CONFIDENCE=0.25 `
  --restart unless-stopped `
  basim123/chickens-wings-counter:cpu
```

Open **http://localhost:5580** in your browser.

> **Note:** CPU inference is significantly slower than GPU. Video and stream processing will run at reduced FPS.

---

#### RTSP stream (optional)

Pass your camera URL via the `-e RTSP_URL=` flag:

```bash
docker run -d --gpus all -p 5580:5580 \
  -e RTSP_URL=rtsp://user:pass@192.168.1.100:554/stream \
  basim123/chickens-wings-counter:latest
```

---

#### Useful container commands

```bash
docker logs -f wing-counter          # tail logs
docker stop wing-counter             # stop
docker rm wing-counter               # remove
docker pull basim123/chickens-wings-counter:latest && \
  docker stop wing-counter && docker rm wing-counter && \
  docker run ...                     # update to latest
```

### Option 3: Cloud deployment

#### AWS EC2 / GCP Compute Engine

1. Launch a GPU instance (e.g., `g4dn.xlarge` on AWS, `n1-standard-4` + T4 on GCP)
2. Install NVIDIA drivers and Docker
3. Clone the project and copy `best.pt` into the root
4. Run `docker compose up -d`
5. Open port **5580** in the security group / firewall

#### Behind a reverse proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5580;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;           # Required for MJPEG streaming
        proxy_cache off;
        proxy_read_timeout 3600s;      # Keep stream connections alive
    }
}
```

---

## Configuration

All settings can be configured via environment variables or the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `best.pt` | Path to YOLOv8 model weights |
| `RTSP_URL` | (empty) | Default RTSP stream URL |
| `ROI_POSITION` | `0.7` | ROI line position (0.0 = top, 1.0 = bottom) |
| `CONFIDENCE` | `0.25` | Detection confidence threshold |
| `MAX_DISTANCE` | `40` | Max pixel distance for tracker matching |
| `MAX_DISAPPEARED` | `50` | Frames before dropping a lost track |

Settings can also be updated at runtime via the `/api/config` endpoint or the stream page UI sliders.

---

## API Reference

### Image Detection

```
POST /api/image/detect
Content-Type: multipart/form-data
Body: file=<image>

Response: image/jpeg (annotated image)
Headers: X-Wing-Count: <number>
```

### Video Processing

```
POST   /api/video/upload                 # Upload video, returns { session_id }
POST   /api/video/{id}/start             # Start playback
POST   /api/video/{id}/stop              # Stop playback
POST   /api/video/{id}/counting/start    # Enable counting
POST   /api/video/{id}/counting/stop     # Disable counting
GET    /api/video/{id}/feed              # MJPEG stream
GET    /api/video/{id}/status            # { wing_count, frame_num, fps, ... }
GET    /api/video/{id}/download          # Download H.264 output
```

### Live Stream

```
POST   /api/stream/start                 # { url: "rtsp://..." }
POST   /api/stream/stop
POST   /api/stream/counting/start
POST   /api/stream/counting/stop
GET    /api/stream/feed                  # MJPEG stream
GET    /api/stream/status                # { wing_count, fps, is_connected, ... }
```

### Configuration

```
GET    /api/config                       # Get current settings
PUT    /api/config                       # Update settings (partial)
```

---

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (Roboflow)
