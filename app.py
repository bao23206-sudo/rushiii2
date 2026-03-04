import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np

st.set_page_config(page_title="Vehicle Counter", layout="wide")

st.title("🚗 Vehicle Detection & Counting App")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output_streamlit.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    vehicle_count = {cls: 0 for cls in vehicle_classes}
    counted_ids = set()

    line_y = height // 2

    progress = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            iou=0.5,
            verbose=False
        )

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2, obj_id, cls_id, conf) in zip(
                boxes.xyxy[:,0], boxes.xyxy[:,1], boxes.xyxy[:,2], boxes.xyxy[:,3],
                ids, classes, confs
            ):
                label = model.names[cls_id]
                color = (0, 255, 0)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                center_y = int((y1 + y2) / 2)

                if label in vehicle_classes:
                    if obj_id not in counted_ids and center_y > line_y:
                        vehicle_count[label] += 1
                        counted_ids.add(obj_id)

        # Draw counting line
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

        total = sum(vehicle_count.values())

        # Overlay count box
        overlay = frame.copy()
        cv2.rectangle(overlay, (width-260, 20), (width-20, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"Total: {total}", (width-240, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        y_offset = 80
        for cls, cnt in vehicle_count.items():
            cv2.putText(frame, f"{cls}: {cnt}",
                        (width-240, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_offset += 25

        out.write(frame)

        # Hiển thị frame realtime
        stframe.image(frame, channels="BGR")

        processed_frames += 1
        progress.progress(min(processed_frames / frame_count, 1.0))

    cap.release()
    out.release()

    st.success("✅ Processing Completed!")

    st.subheader("📊 Final Count")
    st.write(vehicle_count)

    with open(output_path, "rb") as f:
        st.download_button(
            "⬇ Download Processed Video",
            f,
            file_name="vehicle_output.mp4"
        )