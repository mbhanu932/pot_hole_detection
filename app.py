import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import os
import glob
import time
import torch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pothole Detection & Tracking",
    page_icon="🚧",
    layout="wide"
)

st.title("🚧 Pothole Detection & Tracking")
st.write("AI-based pothole detection using a custom YOLO model.")


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"


# ============================================================
# SIDEBAR - MODEL SETTINGS
# ============================================================

st.sidebar.header("⚙️ Model Controls")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05
)

enable_tta = st.sidebar.checkbox(
    "Enable TTA for Images",
    value=False,
    help="Test Time Augmentation improves detection but is slower."
)


# ============================================================
# SIDEBAR - VIDEO SETTINGS
# ============================================================

st.sidebar.header("🎥 Video Settings")

frame_skip = st.sidebar.selectbox(
    "Process Every Nth Frame",
    options=[1, 2, 3, 4],
    index=1
)

video_imgsz = st.sidebar.selectbox(
    "Video Image Size",
    options=[320, 416, 512, 640],
    index=2
)

max_video_width = st.sidebar.selectbox(
    "Maximum Video Width",
    options=[640, 800, 960, 1280],
    index=2
)


# ============================================================
# HARDWARE INFORMATION
# ============================================================

st.sidebar.header("💻 Hardware")

if torch.cuda.is_available():
    st.sidebar.success(
        f"🚀 GPU: {DEVICE_NAME}"
    )
else:
    st.sidebar.info(
        "💻 CUDA GPU not detected. Using CPU."
    )


# ============================================================
# LOAD CUSTOM POTHOLE MODEL
# ============================================================

@st.cache_resource
def load_pothole_model():

    # --------------------------------------------------------
    # OPTION 1:
    # best.pt in the same folder as app.py
    # --------------------------------------------------------

    root_model = "best.pt"

    if os.path.exists(root_model):
        return YOLO(root_model), root_model


    # --------------------------------------------------------
    # OPTION 2:
    # Search inside runs/
    # --------------------------------------------------------

    search_patterns = [
        "runs/**/best.pt",
        "./runs/**/best.pt"
    ]

    found_weights = []

    for pattern in search_patterns:
        found_weights.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    # Remove duplicates
    found_weights = list(
        dict.fromkeys(found_weights)
    )

    if found_weights:

        # Prefer the latest modified model
        found_weights.sort(
            key=os.path.getmtime,
            reverse=True
        )

        model_path = found_weights[0]

        return YOLO(model_path), model_path


    # --------------------------------------------------------
    # MODEL NOT FOUND
    # --------------------------------------------------------

    raise FileNotFoundError(
        "best.pt was not found. "
        "Place your trained best.pt file "
        "in the same folder as app.py."
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, model_path = load_pothole_model()

    st.sidebar.success(
        "✅ Custom Model Loaded"
    )

    st.sidebar.write(
        f"**Model:** `{model_path}`"
    )

except Exception as e:

    st.error(
        f"❌ Could not load the pothole model:\n\n{e}"
    )

    st.info(
        "Make sure your trained `best.pt` file "
        "is present in the project folder."
    )

    st.stop()


# ============================================================
# SHOW MODEL CLASSES
# ============================================================

st.sidebar.subheader("🔍 Model Classes")

st.sidebar.write(model.names)

class_names = list(model.names.values())


# ============================================================
# CHECK FOR POTHOLE CLASS
# ============================================================

pothole_class_ids = []

for class_id, class_name in model.names.items():

    if "pothole" in str(class_name).lower():

        pothole_class_ids.append(
            int(class_id)
        )


if len(pothole_class_ids) > 0:

    st.sidebar.success(
        "🕳️ Pothole class detected!"
    )

else:

    st.sidebar.error(
        "❌ Pothole class NOT found!"
    )

    st.warning(
        "Your `best.pt` does not appear to contain "
        "a pothole class."
    )

    st.write(
        "Model classes:"
    )

    st.write(model.names)

    st.stop()


# ============================================================
# INPUT MODE
# ============================================================

mode = st.radio(
    "Select Input Source",
    ["Image", "Video"],
    horizontal=True
)


# ============================================================
# IMAGE DETECTION
# ============================================================

if mode == "Image":

    st.header("🖼️ Pothole Detection")

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # DISPLAY INPUT
        # ----------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "Original Image"
            )

            st.image(
                image,
                width="stretch"
            )


        # ----------------------------------------------------
        # RUN YOLO
        # ----------------------------------------------------

        with st.spinner(
            "🔍 Detecting potholes..."
        ):

            results = model.predict(

                source=image,

                conf=confidence_threshold,

                iou=iou_threshold,

                imgsz=640,

                augment=enable_tta,

                device=DEVICE,

                verbose=False
            )


        result = results[0]


        # ----------------------------------------------------
        # ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated_image = result.plot()


        with col2:

            st.subheader(
                "Detected Potholes"
            )

            st.image(
                annotated_image,
                width="stretch"
            )


        # ----------------------------------------------------
        # DETECTION COUNT
        # ----------------------------------------------------

        if result.boxes is not None:

            num_detections = len(
                result.boxes
            )

        else:

            num_detections = 0


        st.divider()


        metric1, metric2 = st.columns(2)


        with metric1:

            st.metric(
                "🕳️ Potholes Detected",
                num_detections
            )


        with metric2:

            if num_detections > 0:

                st.success(
                    "Potholes detected!"
                )

            else:

                st.info(
                    "No potholes detected."
                )


        # ----------------------------------------------------
        # DETECTION DETAILS
        # ----------------------------------------------------

        if num_detections > 0:

            st.subheader(
                "🔎 Detection Details"
            )


            for i, box in enumerate(
                result.boxes
            ):

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = model.names[
                    class_id
                ]


                st.write(
                    f"**Detection {i + 1}:** "
                    f"{class_name} — "
                    f"Confidence: "
                    f"{confidence:.2%}"
                )


# ============================================================
# VIDEO DETECTION AND TRACKING
# ============================================================

else:

    st.header(
        "🎥 Pothole Detection & Tracking"
    )


    uploaded_video = st.file_uploader(

        "Upload a road video",

        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )


    if uploaded_video is not None:

        temp_video_path = None


        try:

            # ------------------------------------------------
            # SAVE UPLOADED VIDEO
            # ------------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(
                    uploaded_video.read()
                )

                temp_video_path = (
                    temp_file.name
                )


            # ------------------------------------------------
            # OPEN VIDEO
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                temp_video_path
            )


            if not cap.isOpened():

                st.error(
                    "❌ Could not open video."
                )

                st.stop()


            # ------------------------------------------------
            # VIDEO INFORMATION
            # ------------------------------------------------

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )


            original_fps = cap.get(
                cv2.CAP_PROP_FPS
            )


            if original_fps <= 0:

                original_fps = 30.0


            video_width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )


            video_height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )


            if total_frames > 0:

                video_duration = (
                    total_frames /
                    original_fps
                )

            else:

                video_duration = 0


            # ------------------------------------------------
            # VIDEO INFORMATION DISPLAY
            # ------------------------------------------------

            info1, info2, info3, info4 = (
                st.columns(4)
            )


            info1.metric(
                "Duration",
                f"{video_duration:.1f}s"
            )


            info2.metric(
                "FPS",
                f"{original_fps:.1f}"
            )


            info3.metric(
                "Resolution",
                f"{video_width} × {video_height}"
            )


            info4.metric(
                "Frames",
                total_frames
            )


            st.divider()


            st.subheader(
                "🚀 Processing Video"
            )


            st.write(
                f"Running on: **{DEVICE_NAME}**"
            )


            if DEVICE == "cpu":

                st.caption(
                    "CPU mode detected. "
                    "Frame skipping and resized "
                    "frames are being used."
                )


            # ------------------------------------------------
            # STREAMING FRAME
            # ------------------------------------------------

            video_display = st.empty()


            # ------------------------------------------------
            # PROGRESS BAR
            # ------------------------------------------------

            progress_bar = st.progress(
                0,
                text="Starting video processing..."
            )


            # ------------------------------------------------
            # LIVE METRICS
            # ------------------------------------------------

            metric1, metric2, metric3 = (
                st.columns(3)
            )


            current_frame_metric = (
                metric1.empty()
            )

            track_metric = (
                metric2.empty()
            )

            speed_metric = (
                metric3.empty()
            )


            # ------------------------------------------------
            # TRACKING VARIABLES
            # ------------------------------------------------

            unique_track_ids = set()

            frame_number = 0

            processed_frames = 0

            processing_start = time.time()


            # ------------------------------------------------
            # VIDEO LOOP
            # ------------------------------------------------

            while cap.isOpened():

                ret, frame = cap.read()


                if not ret:

                    break


                frame_number += 1


                # --------------------------------------------
                # FRAME SKIPPING
                # --------------------------------------------

                if (
                    frame_number %
                    frame_skip != 0
                ):

                    continue


                processed_frames += 1


                # --------------------------------------------
                # RESIZE FRAME
                # --------------------------------------------

                height, width = (
                    frame.shape[:2]
                )


                if width > max_video_width:

                    scale = (
                        max_video_width /
                        width
                    )


                    new_width = int(
                        width * scale
                    )


                    new_height = int(
                        height * scale
                    )


                    frame = cv2.resize(

                        frame,

                        (
                            new_width,
                            new_height
                        ),

                        interpolation=cv2.INTER_AREA
                    )


                # --------------------------------------------
                # YOLO TRACKING
                # --------------------------------------------

                results = model.track(

                    source=frame,

                    conf=confidence_threshold,

                    iou=iou_threshold,

                    imgsz=video_imgsz,

                    persist=True,

                    tracker="botsort.yaml",

                    device=DEVICE,

                    verbose=False
                )


                result = results[0]


                # --------------------------------------------
                # CURRENT FRAME DETECTIONS
                # --------------------------------------------

                frame_potholes = 0


                if (
                    result.boxes is not None
                    and
                    len(result.boxes) > 0
                ):


                    # ----------------------------------------
                    # FILTER ONLY POTHOLES
                    # ----------------------------------------

                    pothole_boxes = []


                    for box in result.boxes:

                        class_id = int(
                            box.cls[0].item()
                        )


                        if (
                            class_id
                            in pothole_class_ids
                        ):

                            pothole_boxes.append(
                                box
                            )


                    frame_potholes = len(
                        pothole_boxes
                    )


                    # ----------------------------------------
                    # TRACK IDS
                    # ----------------------------------------

                    if (
                        result.boxes.id
                        is not None
                    ):

                        track_ids = (
                            result.boxes.id
                            .int()
                            .cpu()
                            .tolist()
                        )


                        for track_id in track_ids:

                            unique_track_ids.add(
                                track_id
                            )


                # --------------------------------------------
                # DRAW RESULTS
                # --------------------------------------------

                annotated_frame = (
                    result.plot()
                )


                # --------------------------------------------
                # CONVERT BGR → RGB
                # --------------------------------------------

                rgb_frame = cv2.cvtColor(

                    annotated_frame,

                    cv2.COLOR_BGR2RGB
                )


                # --------------------------------------------
                # DISPLAY FRAME
                # --------------------------------------------

                video_display.image(

                    rgb_frame,

                    width="stretch"
                )


                # --------------------------------------------
                # UPDATE METRICS
                # --------------------------------------------

                current_frame_metric.metric(

                    "🕳️ Potholes in Frame",

                    frame_potholes
                )


                track_metric.metric(

                    "🎯 Track IDs",

                    len(unique_track_ids)
                )


                # --------------------------------------------
                # PROCESSING SPEED
                # --------------------------------------------

                elapsed = (
                    time.time() -
                    processing_start
                )


                if elapsed > 0:

                    processing_fps = (
                        processed_frames /
                        elapsed
                    )

                else:

                    processing_fps = 0


                speed_metric.metric(

                    "⚡ Processing FPS",

                    f"{processing_fps:.1f}"
                )


                # --------------------------------------------
                # PROGRESS
                # --------------------------------------------

                if total_frames > 0:

                    progress = min(

                        frame_number /
                        total_frames,

                        1.0
                    )

                else:

                    progress = 0


                processed_video_time = (

                    frame_number /
                    original_fps
                )


                progress_bar.progress(

                    progress,

                    text=(

                        f"Processing "
                        f"{progress * 100:.1f}% | "

                        f"Video: "
                        f"{processed_video_time:.1f}s / "
                        f"{video_duration:.1f}s"
                    )
                )


            # ------------------------------------------------
            # RELEASE VIDEO
            # ------------------------------------------------

            cap.release()


            processing_time = (
                time.time() -
                processing_start
            )


            # ------------------------------------------------
            # FINAL RESULTS
            # ------------------------------------------------

            st.divider()


            st.success(
                "✅ Video processing completed!"
            )


            result1, result2, result3 = (
                st.columns(3)
            )


            result1.metric(

                "🎯 Unique Track IDs",

                len(unique_track_ids)
            )


            result2.metric(

                "⏱️ Processing Time",

                f"{processing_time:.1f}s"
            )


            if processing_time > 0:

                average_fps = (

                    processed_frames /
                    processing_time
                )

            else:

                average_fps = 0


            result3.metric(

                "⚡ Average FPS",

                f"{average_fps:.1f}"
            )


            # ------------------------------------------------
            # SPEED COMPARISON
            # ------------------------------------------------

            if video_duration > 0:

                if (
                    processing_time <
                    video_duration
                ):

                    realtime_multiplier = (

                        video_duration /
                        processing_time
                    )


                    st.info(

                        f"Processing speed: "
                        f"{realtime_multiplier:.2f}× "
                        f"the video duration."
                    )

                else:

                    slowdown = (

                        processing_time /
                        video_duration
                    )


                    st.info(

                        f"Processing took "
                        f"{slowdown:.2f}× "
                        f"the video duration."
                    )


        except Exception as e:

            st.error(
                f"❌ Error processing video: {e}"
            )


        finally:

            if (
                temp_video_path
                is not None
                and
                os.path.exists(
                    temp_video_path
                )
            ):

                try:

                    os.remove(
                        temp_video_path
                    )

                except Exception:

                    pass
