# People_Counter

Real-time people counting system using YOLO11n and BoT-SORT with ReID to detect, track, and uniquely count pedestrians (class 0) without double-counting. Built with a Streamlit UI supporting both uploaded video files and live webcam feeds for instant occupancy and cumulative footfall tracking.

## Features

- **Person-Only Filtering:** Strictly isolates COCO `class 0` (`person`), ignoring background objects.
- **Persistent Multi-Object Tracking:** Combines BoT-SORT motion modeling (`sparseOptFlow` camera motion compensation) with ReID appearance embeddings to maintain consistent Track IDs.
- **Dual Input Modes:**
  1. **Upload Video:** Process `.mp4`, `.avi`, `.mov`, and `.mkv` files with real-time bounding box overlays.
  2. **Live Feed:** Stream directly from a local webcam (`cv2.VideoCapture`).
- **Interactive Controls:** Adjust detection confidence thresholds dynamically from the sidebar.

## Project Structure

```text
├── app.py                # Streamlit frontend and tracking pipeline
├── botsort_reid.yaml     # Custom BoT-SORT + ReID configuration
├── yolo11n.pt            # Pre-trained YOLO11n weights
├── requirements.txt      # Python dependencies
└── README.md
```

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install dependencies** (Python 3.10–3.12 recommended)

   python -m venv .venv
   source .venv/bin/activate

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
