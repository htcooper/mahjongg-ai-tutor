import tkinter as tk
from tkinter import ttk
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Model helpers (copied from object_detection.py to avoid side-effects)
# ---------------------------------------------------------------------------

def load_graph(model_file: str) -> tf.Graph:
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name="")
    return graph


def load_labels(labels_path: str) -> dict[int, dict]:
    with open(labels_path, "r") as f:
        labels = f.read().splitlines()
    return {i: {"id": i, "name": label} for i, label in enumerate(labels)}


def run_inference(session, image_tensor, det_boxes, det_scores, det_classes, image):
    outputs = session.run(
        [det_boxes, det_scores, det_classes],
        feed_dict={image_tensor: image},
    )
    return {
        "detection_boxes": np.squeeze(outputs[0]),
        "detection_scores": np.squeeze(outputs[1]),
        "detection_classes": np.squeeze(outputs[2]).astype(np.int32),
    }


def draw_boxes(
    image, boxes, scores, classes, category_index, min_score_thresh: float = 0.3
):
    height, width, _ = image.shape
    for i in range(len(scores)):
        if scores[i] >= min_score_thresh:
            x1, y1, x2, y2 = boxes[i]
            left, right = int(x1 * width), int(x2 * width)
            top, bottom = int(y1 * height), int(y2 * height)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            class_index = classes[i]
            label = category_index.get(class_index, {}).get("name", "Unknown")
            text = f"{label}: {scores[i]:.2f}"
            cv2.putText(
                image, text, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2,
            )
    return image


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model.pb")
LABELS_PATH = str(BASE_DIR / "labels.txt")


class MahjongDetectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mahjong Tile Detector")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.cap: cv2.VideoCapture | None = None
        self.running = False

        # ---- Load model ----
        self.category_index = load_labels(LABELS_PATH)
        self.graph = load_graph(MODEL_PATH)
        self.session = tf.compat.v1.Session(graph=self.graph)
        with self.graph.as_default():
            self.image_tensor = self.graph.get_tensor_by_name("image_tensor:0")
            self.det_boxes = self.graph.get_tensor_by_name("detected_boxes:0")
            self.det_scores = self.graph.get_tensor_by_name("detected_scores:0")
            self.det_classes = self.graph.get_tensor_by_name("detected_classes:0")

        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self) -> None:
        # Video feed
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(padx=10, pady=(10, 5))

        # Controls frame
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=10, pady=5)

        self.start_btn = ttk.Button(ctrl, text="Start", command=self._start)
        self.start_btn.pack(side="left", padx=(0, 5))

        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(0, 15))

        ttk.Label(ctrl, text="Camera:").pack(side="left")
        self.camera_var = tk.StringVar(value="0")
        cam_dropdown = ttk.Combobox(
            ctrl, textvariable=self.camera_var, values=["0", "1", "2", "3", "4"],
            width=3, state="readonly",
        )
        cam_dropdown.pack(side="left", padx=(5, 15))

        ttk.Label(ctrl, text="Confidence:").pack(side="left")
        self.threshold_var = tk.DoubleVar(value=0.30)
        self.threshold_slider = ttk.Scale(
            ctrl, from_=0.05, to=0.95, variable=self.threshold_var,
            orient="horizontal", length=200, command=self._on_threshold_change,
        )
        self.threshold_slider.pack(side="left", padx=(5, 5))
        self.threshold_label = ttk.Label(ctrl, text="0.30")
        self.threshold_label.pack(side="left")

        # Detected tiles panel
        det_frame = ttk.LabelFrame(self.root, text="Detected Tiles")
        det_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        self.detections_text = tk.Text(
            det_frame, height=4, wrap="word", state="disabled",
            font=("Consolas", 10),
        )
        scrollbar = ttk.Scrollbar(det_frame, command=self.detections_text.yview)
        self.detections_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.detections_text.pack(fill="both", expand=True)

        # Placeholder image
        placeholder = Image.new("RGB", (640, 480), (30, 30, 30))
        self._photo = ImageTk.PhotoImage(placeholder)
        self.video_label.configure(image=self._photo)

    # ---- Letterbox (preserve aspect ratio) ----

    @staticmethod
    def _letterbox(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    # ---- Threshold slider callback ----

    def _on_threshold_change(self, _event=None) -> None:
        self.threshold_label.configure(text=f"{self.threshold_var.get():.2f}")

    # ---- Start / stop ----

    def _start(self) -> None:
        cam_index = int(self.camera_var.get())
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            self._update_detections_text("Error: Could not open camera.")
            return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._update_frame()

    def _stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    # ---- Frame loop ----

    def _update_frame(self) -> None:
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return

        threshold = self.threshold_var.get()

        # Run inference — convert BGR->RGB and letterbox to preserve aspect ratio
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        scale = min(320 / w, 320 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        x_off = (320 - new_w) / 2
        y_off = (320 - new_h) / 2

        resized = self._letterbox(rgb_frame, 320, 320)
        input_frame = np.expand_dims(resized, axis=0).astype(np.float32)
        output = run_inference(
            self.session, self.image_tensor,
            self.det_boxes, self.det_scores, self.det_classes, input_frame,
        )

        # Remap boxes from letterboxed 320x320 space to original frame space
        boxes = output["detection_boxes"].copy()
        # boxes are [x1, y1, x2, y2] normalized to 320x320
        boxes[:, 0] = (boxes[:, 0] * 320 - x_off) / new_w  # x1
        boxes[:, 1] = (boxes[:, 1] * 320 - y_off) / new_h  # y1
        boxes[:, 2] = (boxes[:, 2] * 320 - x_off) / new_w  # x2
        boxes[:, 3] = (boxes[:, 3] * 320 - y_off) / new_h  # y2
        boxes = np.clip(boxes, 0.0, 1.0)
        output["detection_boxes"] = boxes

        # Draw boxes on original frame
        frame_drawn = draw_boxes(
            frame, output["detection_boxes"], output["detection_scores"],
            output["detection_classes"], self.category_index,
            min_score_thresh=threshold,
        )

        # Update detections panel
        self._update_detections(output, threshold)

        # Convert to Tkinter image
        rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480))
        self._photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self._photo)

        # Schedule next frame (~30 fps)
        self.root.after(33, self._update_frame)

    # ---- Detection panel ----

    def _update_detections(self, output: dict, threshold: float) -> None:
        scores = output["detection_scores"]
        classes = output["detection_classes"]
        counts: Counter = Counter()
        for i in range(len(scores)):
            if scores[i] >= threshold:
                cls = classes[i]
                name = self.category_index.get(cls, {}).get("name", "Unknown")
                counts[name] += 1

        if counts:
            parts = [f"{name}: {count}" for name, count in sorted(counts.items())]
            text = "  |  ".join(parts)
        else:
            text = "(no detections)"
        self._update_detections_text(text)

    def _update_detections_text(self, text: str) -> None:
        self.detections_text.configure(state="normal")
        self.detections_text.delete("1.0", "end")
        self.detections_text.insert("1.0", text)
        self.detections_text.configure(state="disabled")

    # ---- Cleanup ----

    def _on_closing(self) -> None:
        self._stop()
        self.session.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = MahjongDetectorApp(root)
    root.mainloop()
