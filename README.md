# People_Counter

Real-time people counting system using YOLO11n and BoT-SORT with ReID to detect, track, and uniquely count pedestrians (class 0) without double-counting. Built with a Streamlit UI supporting both uploaded video files and live webcam feeds for instant occupancy and cumulative footfall tracking.

## Features

- **Person-Only Filtering:** Strictly isolates COCO `class 0` (`person`), ignoring background objects.
- **Persistent Multi-Object Tracking:** Combines BoT-SORT motion modeling (`sparseOptFlow` camera motion compensation) with ReID appearance embeddings to maintain consistent Track IDs.
- **Dual Input Modes:**
  1. **Upload Video:** Process `.mp4`, `.avi`, `.mov`, and `.mkv` files with real-time bounding box overlays.
  2. **Live Feed:** Stream directly from a local webcam (`cv2.VideoCapture`).
- **Interactive Controls:** Adjust detection confidence thresholds dynamically from the sidebar.

## Quick Start

1. **Create venv** <br>
    python -m venv .venv  <br>
    source .venv/bin/activate  <br>
   
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
