import os
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="People Counter", layout="wide")
st.title("People Counting System")

WEIGHTS_PATH = "yolo11n.pt"
TRACKER_CONFIG = "botsort_reid.yaml"

# check whether weights and tracker config exist
if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(TRACKER_CONFIG):
    st.error(f"Missing `{WEIGHTS_PATH}` or `{TRACKER_CONFIG}` in the workspace folder.")
    st.stop()

#With caching, the model is loaded once and reused.
@st.cache_resource
def load_model():
    return YOLO(WEIGHTS_PATH)

model = load_model()

# Sidebar Controls
st.sidebar.header("Tracking Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.00, 0.30, 0.05)  #	Increment step.

# cap =video capture object,
#
#
def process_stream(cap, stop_key: str):
    # Reset tracker state so Track IDs start cleanly at 1 for every new video or live feed
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None

    # Centered Stop button
    _, mid_col, _ = st.columns([2, 1, 2])
    stop_btn = mid_col.button("Stop Stream", key=stop_key, width="stretch")

    # Full-width metrics and video output
    col1, col2 = st.columns(2)

    current_metric = col1.empty()
    unique_metric = col2.empty()
    frame_placeholder = st.empty()

#Creating a Set for Unique People
    unique_person_ids = set()

#Starting the Video Processing Loop
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO11 + BoT-SORT + ReID strictly on 'person' (class 0)
        results = model.track(
            frame,
            classes=[0],
            conf=conf_threshold,
            persist=True, #tells the tracker to link detections across sequential frames rather than treating each frame independently.
            tracker=TRACKER_CONFIG,
            verbose=False
        )

        annotated_frame = frame.copy()
        boxes = results[0].boxes
        current_people = 0

        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.int().cpu().tolist() #box dimension
            cls_list = boxes.cls.int().cpu().tolist() #class
            conf_list = boxes.conf.cpu().tolist() #confidence
            id_list = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(cls_list)


            for box, cls_id, conf, track_id in zip(xyxy_list, cls_list, conf_list, id_list):
                # Hard filter: strictly ignore any detection that is not class 0 (person)
                if cls_id != 0:
                    continue

                current_people += 1
                x1, y1, x2, y2 = box

                if track_id is not None:
                    unique_person_ids.add(track_id)
                    label = f"Person ID:{track_id} ({conf:.2f})"
                else:
                    label = f"Person ({conf:.2f})"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Draw count banner onto the video frame
        cv2.putText(
            annotated_frame,
            f"In Frame: {current_people} | Total Unique: {len(unique_person_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3
        )

        current_metric.metric("People in Current Frame", current_people)
        unique_metric.metric("Total Unique People (ReID)", len(unique_person_ids))
        frame_placeholder.image(annotated_frame, channels="BGR", width="stretch")

    cap.release()

# --- TWO MAIN UI FEATURES ---
tab_upload, tab_live = st.tabs(["1. Upload Video", "2. Live Feed"])

with tab_upload:
    st.subheader("Count People from Uploaded Video")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        # Center only the button; keep process_stream() outside so video renders full-width
        _, mid_col1, _ = st.columns([2, 1, 2])
        start_video = mid_col1.button("Start Counting Video", key="start_video", width="stretch")

        if start_video:
            cap = cv2.VideoCapture(tfile.name)
            process_stream(cap, stop_key="stop_video")

with tab_live:
    st.subheader("Count People from Live Camera")
    cam_index = st.number_input("Camera Index (0 = Default Webcam)", min_value=0, max_value=5, value=0, step=1)

    _, mid_col2, _ = st.columns([2, 1, 2])
    start_live = mid_col2.button("Start Live Feed", key="start_live", width="stretch")

    if start_live:
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(f"Could not access camera at index {cam_index}. Check permissions or try index 1.")
        else:
            process_stream(cap, stop_key="stop_live")
