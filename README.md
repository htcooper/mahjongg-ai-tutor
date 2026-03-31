# Mahjong AI Tutor

Real-time Mahjong tile detection using a TensorFlow model exported from [Azure Custom Vision](https://www.customvision.ai/). Point a webcam at Mahjong tiles and the app identifies and counts them in real time.

## Features

- **Real-time detection** — runs inference on a live webcam feed at ~30 fps
- **Adjustable confidence threshold** — slider to tune detection sensitivity (5%–95%)
- **Tile counting** — aggregated count of each detected tile type displayed in a side panel
- **Multi-camera support** — select from camera indices 0–4
- **43 tile classes** — winds, dragons, dots, bams, cracks, flowers, and joker

## Prerequisites

- Python 3.6+
- TensorFlow 2.1+ (`pip install tensorflow`)
- OpenCV (`pip install opencv-python`)
- Pillow (`pip install pillow`)
- NumPy (`pip install numpy`)
- tkinter (bundled with most Python installations)

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/mahjongg-ai-tutor.git
   cd mahjongg-ai-tutor
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python pillow numpy
   ```

3. Place your `model.pb` file in the project root. This is a frozen graph exported from Azure Custom Vision (General Compact domain). The file is not included in the repo due to size.

## Usage

```bash
python app.py
```

1. Select a camera index and click **Start** to begin the video feed.
2. Adjust the **confidence threshold** slider to filter out low-confidence detections.
3. Detected tiles appear with bounding boxes and labels overlaid on the video.
4. The right panel shows a running count of each tile type currently detected.
5. Click **Stop** to pause detection.

## Project Structure

| File | Description |
|---|---|
| `app.py` | Main GUI application (Tkinter) — video capture, inference, and display |
| `object_detection.py` | Utility functions for model loading, inference, and drawing bounding boxes |
| `inspect_graph.py` | Diagnostic tool to list all operations in the .pb model graph |
| `labels.txt` | 43 Mahjong tile class labels |
| `model.pb` | Frozen TensorFlow graph (not included — export from Azure Custom Vision) |

## Model Details

- **Source**: Azure Custom Vision (General Compact domain, exported as frozen TensorFlow graph)
- **Input**: 320x320 RGB image (letterboxed to preserve aspect ratio)
- **Outputs**:
  - `detected_boxes` — bounding box coordinates `[x1, y1, x2, y2]` (normalized 0–1)
  - `detected_scores` — confidence probability per detection
  - `detected_classes` — class index per detection (maps to `labels.txt`)
- **Max detections per frame**: 64

## Supported Tile Classes

| Category | Tiles |
|---|---|
| **Winds** | East, South, West, North |
| **Dragons** | Red, Green, Soap (White) |
| **Dots** | 1–9 |
| **Bams** (Bamboo) | 1–9 |
| **Cracks** (Characters) | 1–9 |
| **Flowers** | Spring, Summer, Autumn, Winter, 1–3, 4 |
| **Other** | Joker |
