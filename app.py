import glob
import os
import tempfile
import time
import cv2
import numpy as np
import torch
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="High-Accuracy Pothole Detection & Tracking",
    page_icon="🚧",
    layout="wide",
)

st.title("🚧 Real-Time Video Pothole Detection & Tracking")

# Initialize Session State for Video Processing Control
if "stop_processing" not in st.session_state:
    st.session_state.stop_processing = False

# ============================================================
# AUTOMATIC DEVICE DETECTION
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"

# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("⚙️ Model Controls & Settings")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05,
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
)

tracker_type = st.sidebar.selectbox(
    "Tracking Algorithm",
    options=["bytetrack.yaml", "botsort.yaml"],
    index=0,
    help="ByteTrack utilizes LAP matching efficiently. BoT-SORT adds camera motion compensation.",
)

enable_tta = st.sidebar.checkbox(
    "Enable TTA for Images",
    value=False,
    help="Test-Time Augmentation improves image accuracy but slows down inference.",
)

# ============================================================
# VIDEO PERFORMANCE SETTINGS
# ============================================================

st.sidebar.header("🚀 Video Performance")

frame_skip = st.sidebar.selectbox(
    "Process Every Nth Frame",
    options=[1, 2, 3, 4],
    index=1,
    help="1 = process all frames, 2 = skip every other frame (faster).",
)

video_imgsz = st.sidebar.selectbox(
    "Video Inference Size",
    options=[320, 416, 512, 640],
    index=2,
    help="Smaller resolution leads to faster processing, especially on CPU.",
)

max_video_width = st.sidebar.selectbox(
    "Maximum Display Width",
    options=[640, 800, 960, 1280],
    index=2,
    help="Resizes raw frame before inference to optimize memory and speed.",
)

# ============================================================
# HARDWARE INFORMATION
# ============================================================

st.sidebar.header("💻 Hardware Status")
if torch.cuda.is_available():
    st.sidebar.success(f"🚀 {DEVICE_NAME}")
else:
    st.sidebar.info("💻 CUDA GPU not detected. Running on CPU.")

# ============================================================
# MODEL LOADING FUNCTION
# ============================================================


@st.cache_resource
def load_pothole_model():
    # 1. Custom model in app directory
    if os.path.exists("best.pt"):
        return YOLO("best.pt")

    # 2. Search recursively in runs directory
    found_weights = glob.glob("runs/**/best.pt", recursive=True)
    if found_weights:
        return YOLO(found_weights[0])

    # 3. Fallback pretrained YOLO model
    if os.path.exists("yolov8n.pt"):
        st.sidebar.warning(
            "⚠️ Using base YOLOv8n model. For accurate pothole detection, provide trained `best.pt` weights."
        )
        return YOLO("yolov8n.pt")

    raise FileNotFoundError(
        "No model weights found. Please place `best.pt` in the application root."
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

try:
    model = load_pothole_model()
    st.sidebar.success("✅ Model Loaded Successfully!")
    st.sidebar.write("**Model Classes:**", model.names)
except Exception as e:
    st.error(f"❌ Failed to load model weights: {e}")
    st.stop()

# ============================================================
# INPUT MODE SELECTION
# ============================================================

mode = st.radio("Select Input Source:", ("Image", "Video"), horizontal=True)

# ============================================================
# IMAGE DETECTION MODE
# ============================================================

if mode == "Image":
    st.header("🖼️ Image Pothole Detection")

    uploaded_file = st.file_uploader(
        "Upload Road Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Input")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🔍 Predictions")
            with st.spinner("Analyzing image..."):
                results = model.predict(
                    source=image,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    imgsz=640,
                    augment=enable_tta,
                    device=DEVICE,
                    verbose=False,
                )

            result = results[0]
            annotated_image = result.plot()
            annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            st.image(
                annotated_rgb,
                caption="Detected Potholes",
                use_container_width=True,
            )

            num_detected = (
                len(result.boxes) if result.boxes is not None else 0
            )
            st.metric("🕳️ Total Potholes Detected", num_detected)

# ============================================================
# VIDEO DETECTION & TRACKING MODE
# ============================================================

elif mode == "Video":
    st.header("🎥 Video Pothole Detection & Tracking")

    uploaded_video = st.file_uploader(
        "Upload Road Video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        temp_input_path = None
        temp_output_path = None

        try:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as tfile:
                tfile.write(uploaded_video.read())
                temp_input_path = tfile.name

            cap = cv2.VideoCapture(temp_input_path)
            if not cap.isOpened():
                st.error("❌ Error opening video file.")
                st.stop()

            # Video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps <= 0 or np.isnan(original_fps):
                original_fps = 30.0

            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = (
                total_frames / original_fps if total_frames > 0 else 0
            )

            # Display Metadata
            info1, info2, info3, info4 = st.columns(4)
            info1.metric("Duration", f"{video_duration:.1f}s")
            info2.metric("FPS", f"{original_fps:.1f}")
            info3.metric("Resolution", f"{video_width}×{video_height}")
            info4.metric("Total Frames", total_frames)

            st.divider()

            # Output Video setup
            output_temp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            )
            temp_output_path = output_temp.name
            output_temp.close()

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = None

            # Streamlit UI placeholders
            st.subheader("🚀 Processing Video")
            st.caption(f"Running inference engine on **{DEVICE_NAME}**")

            st_frame = st.empty()
            progress_bar = st.progress(0, text="Initializing tracker...")

            mcol1, mcol2, mcol3 = st.columns(3)
            current_metric = mcol1.empty()
            total_metric = mcol2.empty()
            speed_metric = mcol3.empty()

            # Stop Processing Control Button
            if st.button("⛔ Stop Processing"):
                st.session_state.stop_processing = True

            unique_pothole_ids = set()
            frame_number = 0
            processed_frames = 0
            processing_start = time.time()
            st.session_state.stop_processing = False

            # Processing Loop
            while cap.isOpened() and not st.session_state.stop_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                # Frame Skipping Logic
                if frame_number % frame_skip != 0:
                    continue

                processed_frames += 1

                # Resize Frame if width exceeds limits
                h, w = frame.shape[:2]
                if w > max_video_width:
                    scale = max_video_width / w
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )

                # Initialize VideoWriter on first valid frame processing
                if out_writer is None:
                    out_h, out_w = frame.shape[:2]
                    out_writer = cv2.VideoWriter(
                        temp_output_path,
                        fourcc,
                        original_fps / frame_skip,
                        (out_w, out_h),
                    )

                # Run Object Tracking with LAP matching
                results = model.track(
                    source=frame,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    imgsz=video_imgsz,
                    persist=True,
                    tracker=tracker_type,
                    device=DEVICE,
                    verbose=False,
                )

                result = results[0]
                frame_potholes = 0

                # ID extraction
                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    track_ids = result.boxes.id.int().cpu().tolist()
                    for t_id in track_ids:
                        unique_pothole_ids.add(t_id)
                    frame_potholes = len(track_ids)
                elif result.boxes is not None:
                    frame_potholes = len(result.boxes)

                # Annotate Frame
                annotated_frame = result.plot()
                out_writer.write(annotated_frame)

                # Display to Streamlit (Convert BGR to RGB)
                rgb_frame = cv2.cvtColor(
                    annotated_frame, cv2.COLOR_BGR2RGB
                )
                st_frame.image(rgb_frame, use_container_width=True)

                # Update live metrics
                elapsed = time.time() - processing_start
                processing_fps = (
                    (processed_frames / elapsed) if elapsed > 0 else 0
                )

                current_metric.metric("Potholes (Current)", frame_potholes)
                total_metric.metric(
                    "Unique Tracked", len(unique_pothole_ids)
                )
                speed_metric.metric(
                    "Processing FPS", f"{processing_fps:.1f}"
                )

                # Update Progress Bar
                progress = min(frame_number / total_frames, 1.0)
                progress_bar.progress(
                    progress,
                    text=f"Processed: {frame_number}/{total_frames} frames ({progress * 100:.1f}%)",
                )

            # Clean Video Operations
            cap.release()
            if out_writer is not None:
                out_writer.release()

            total_processing_time = time.time() - processing_start
            st.divider()

            if st.session_state.stop_processing:
                st.warning("⛔ Video processing stopped by user.")
            else:
                st.success("✅ Video processing completed!")

            # Final Metrics
            res1, res2, res3 = st.columns(3)
            res1.metric("Unique Potholes Found", len(unique_pothole_ids))
            res2.metric("Elapsed Time", f"{total_processing_time:.1f}s")
            avg_fps = (
                (processed_frames / total_processing_time)
                if total_processing_time > 0
                else 0
            )
            res3.metric("Average Speed", f"{avg_fps:.1f} FPS")

            # Option to download processed output video
            if os.path.exists(temp_output_path):
                with open(temp_output_path, "rb") as video_file:
                    st.download_button(
                        label="📥 Download Annotated Video",
                        data=video_file,
                        file_name="pothole_tracked_output.mp4",
                        mime="video/mp4",
                    )

        except Exception as e:
            st.error(f"❌ Error during video processing: {e}")

        finally:
            # Temporary file cleanup
            for p in [temp_input_path, temp_output_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass




# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import cv2
# import tempfile
# import os
# import glob
# import time
# import torch
# import math


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Pothole Detection & Tracking",
#     page_icon="🚧",
#     layout="wide"
# )

# st.title("🚧 Pothole Detection & Tracking")

# st.write(
#     "Detect, track and count unique potholes throughout an entire road video."
# )


# # ============================================================
# # DEVICE
# # ============================================================

# if torch.cuda.is_available():
#     DEVICE = 0
#     DEVICE_NAME = torch.cuda.get_device_name(0)
# else:
#     DEVICE = "cpu"
#     DEVICE_NAME = "CPU"


# # ============================================================
# # SIDEBAR
# # ============================================================

# st.sidebar.header("⚙️ Detection Settings")

# confidence_threshold = st.sidebar.slider(
#     "Detection Confidence",
#     0.05,
#     1.0,
#     0.40,
#     0.05
# )

# iou_threshold = st.sidebar.slider(
#     "IoU Threshold",
#     0.10,
#     0.90,
#     0.45,
#     0.05
# )


# # ============================================================
# # VIDEO SETTINGS
# # ============================================================

# st.sidebar.header("🎥 Video Settings")

# video_imgsz = st.sidebar.selectbox(
#     "Video Inference Size",
#     [320, 416, 512, 640],
#     index=2
# )

# max_video_width = st.sidebar.selectbox(
#     "Maximum Video Width",
#     [640, 800, 960, 1280],
#     index=2
# )


# # ============================================================
# # TRACKING SETTINGS
# # ============================================================

# st.sidebar.header("🎯 Tracking Settings")

# min_confirm_frames = st.sidebar.slider(
#     "Minimum Frames to Confirm Pothole",
#     2,
#     15,
#     5
# )

# max_missing_frames = st.sidebar.slider(
#     "Maximum Missing Frames",
#     5,
#     60,
#     20
# )

# spatial_match_distance = st.sidebar.slider(
#     "Pothole Match Distance",
#     20,
#     200,
#     80
# )


# st.sidebar.info(
#     "A pothole must persist for several frames before it is counted. "
#     "This helps prevent duplicate/temporary detections."
# )


# # ============================================================
# # HARDWARE
# # ============================================================

# st.sidebar.header("💻 Hardware")

# if torch.cuda.is_available():

#     st.sidebar.success(
#         f"🚀 GPU: {torch.cuda.get_device_name(0)}"
#     )

# else:

#     st.sidebar.info(
#         "💻 Using CPU"
#     )


# # ============================================================
# # LOAD CUSTOM MODEL
# # ============================================================

# @st.cache_resource
# def load_pothole_model():

#     # First check current directory
#     if os.path.exists("best.pt"):
#         return YOLO("best.pt")

#     # Search runs directory
#     found_weights = glob.glob(
#         "runs/**/best.pt",
#         recursive=True
#     )

#     if found_weights:

#         found_weights.sort(
#             key=os.path.getmtime,
#             reverse=True
#         )

#         return YOLO(
#             found_weights[0]
#         )

#     raise FileNotFoundError(
#         "best.pt was not found. "
#         "Place your custom best.pt in the same folder as app.py."
#     )


# # ============================================================
# # INITIALIZE MODEL
# # ============================================================

# try:

#     model = load_pothole_model()

#     st.sidebar.success(
#         "✅ Custom pothole model loaded"
#     )

#     st.sidebar.write(
#         "**Classes:**",
#         model.names
#     )

# except Exception as e:

#     st.error(
#         f"❌ Model loading failed: {e}"
#     )

#     st.stop()


# # ============================================================
# # FIND POTHOLE CLASS
# # ============================================================

# pothole_class_ids = set()

# for class_id, class_name in model.names.items():

#     if str(class_name).lower() == "pothole":

#         pothole_class_ids.add(
#             int(class_id)
#         )


# if not pothole_class_ids:

#     st.error(
#         "❌ No 'pothole' class was found in the model."
#     )

#     st.write(
#         "Available classes:",
#         model.names
#     )

#     st.stop()


# # ============================================================
# # HELPER FUNCTIONS
# # ============================================================

# def get_box_center(box):

#     x1, y1, x2, y2 = box

#     center_x = (x1 + x2) / 2
#     center_y = (y1 + y2) / 2

#     return center_x, center_y


# def center_distance(center1, center2):

#     return math.sqrt(
#         (center1[0] - center2[0]) ** 2
#         +
#         (center1[1] - center2[1]) ** 2
#     )


# def calculate_iou(box1, box2):

#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])

#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     intersection_width = max(
#         0,
#         x2 - x1
#     )

#     intersection_height = max(
#         0,
#         y2 - y1
#     )

#     intersection = (
#         intersection_width
#         *
#         intersection_height
#     )

#     area1 = (
#         max(0, box1[2] - box1[0])
#         *
#         max(0, box1[3] - box1[1])
#     )

#     area2 = (
#         max(0, box2[2] - box2[0])
#         *
#         max(0, box2[3] - box2[1])
#     )

#     union = (
#         area1
#         +
#         area2
#         -
#         intersection
#     )

#     if union <= 0:
#         return 0

#     return intersection / union


# # ============================================================
# # INPUT MODE
# # ============================================================

# mode = st.radio(
#     "Select Input",
#     ["Image", "Video"],
#     horizontal=True
# )


# # ============================================================
# # IMAGE MODE
# # ============================================================

# if mode == "Image":

#     st.header("🖼️ Pothole Detection")

#     uploaded_file = st.file_uploader(
#         "Upload Road Image",
#         type=["jpg", "jpeg", "png"]
#     )

#     if uploaded_file is not None:

#         image = Image.open(
#             uploaded_file
#         )

#         col1, col2 = st.columns(2)

#         with col1:

#             st.subheader(
#                 "Original Image"
#             )

#             st.image(
#                 image,
#                 use_container_width=True
#             )

#         with col2:

#             st.subheader(
#                 "🔍 Detection Result"
#             )

#             with st.spinner(
#                 "Detecting potholes..."
#             ):

#                 results = model.predict(
#                     source=image,
#                     conf=confidence_threshold,
#                     iou=iou_threshold,
#                     imgsz=1280,
#                     device=DEVICE,
#                     verbose=False
#                 )

#             result = results[0]

#             pothole_count = 0

#             if result.boxes is not None:

#                 for box in result.boxes:

#                     class_id = int(
#                         box.cls[0].item()
#                     )

#                     if class_id in pothole_class_ids:

#                         pothole_count += 1

#             annotated_image = result.plot()

#             st.image(
#                 annotated_image,
#                 use_container_width=True
#             )

#             st.metric(
#                 "🕳️ Potholes Detected",
#                 pothole_count
#             )


# # ============================================================
# # VIDEO MODE
# # ============================================================

# else:

#     st.header(
#         "🎥 Full Video Pothole Detection & Tracking"
#     )

#     uploaded_video = st.file_uploader(
#         "Upload Road Video",
#         type=[
#             "mp4",
#             "avi",
#             "mov",
#             "mkv"
#         ]
#     )

#     if uploaded_video is not None:

#         temp_video_path = None
#         output_video_path = None

#         try:

#             # =================================================
#             # SAVE INPUT VIDEO
#             # =================================================

#             with tempfile.NamedTemporaryFile(
#                 delete=False,
#                 suffix=".mp4"
#             ) as temp_file:

#                 temp_file.write(
#                     uploaded_video.read()
#                 )

#                 temp_video_path = temp_file.name


#             # =================================================
#             # OPEN VIDEO
#             # =================================================

#             cap = cv2.VideoCapture(
#                 temp_video_path
#             )

#             if not cap.isOpened():

#                 st.error(
#                     "❌ Could not open video."
#                 )

#                 st.stop()


#             # =================================================
#             # VIDEO INFORMATION
#             # =================================================

#             total_frames = int(
#                 cap.get(
#                     cv2.CAP_PROP_FRAME_COUNT
#                 )
#             )

#             original_fps = cap.get(
#                 cv2.CAP_PROP_FPS
#             )

#             if original_fps <= 0:
#                 original_fps = 30.0

#             original_width = int(
#                 cap.get(
#                     cv2.CAP_PROP_FRAME_WIDTH
#                 )
#             )

#             original_height = int(
#                 cap.get(
#                     cv2.CAP_PROP_FRAME_HEIGHT
#                 )
#             )

#             video_duration = (
#                 total_frames
#                 /
#                 original_fps
#             )


#             # =================================================
#             # DISPLAY VIDEO INFO
#             # =================================================

#             info1, info2, info3, info4 = st.columns(4)

#             info1.metric(
#                 "Duration",
#                 f"{video_duration:.1f}s"
#             )

#             info2.metric(
#                 "FPS",
#                 f"{original_fps:.1f}"
#             )

#             info3.metric(
#                 "Resolution",
#                 f"{original_width}×{original_height}"
#             )

#             info4.metric(
#                 "Total Frames",
#                 total_frames
#             )


#             st.divider()


#             # =================================================
#             # OUTPUT SIZE
#             # =================================================

#             if original_width > max_video_width:

#                 scale = (
#                     max_video_width
#                     /
#                     original_width
#                 )

#                 output_width = int(
#                     original_width * scale
#                 )

#                 output_height = int(
#                     original_height * scale
#                 )

#             else:

#                 output_width = original_width
#                 output_height = original_height


#             # =================================================
#             # OUTPUT VIDEO
#             # =================================================

#             output_file = tempfile.NamedTemporaryFile(
#                 delete=False,
#                 suffix=".mp4"
#             )

#             output_video_path = output_file.name

#             output_file.close()


#             fourcc = cv2.VideoWriter_fourcc(
#                 *"mp4v"
#             )

#             writer = cv2.VideoWriter(
#                 output_video_path,
#                 fourcc,
#                 original_fps,
#                 (
#                     output_width,
#                     output_height
#                 )
#             )

#             if not writer.isOpened():

#                 raise RuntimeError(
#                     "Could not create output video."
#                 )


#             # =================================================
#             # UI
#             # =================================================

#             st.subheader(
#                 "🚀 Processing Entire Video"
#             )

#             st.write(
#                 f"Running on: **{DEVICE_NAME}**"
#             )

#             st.info(
#                 "Every frame is processed. "
#                 "Repeated detections of the same pothole "
#                 "are merged into one tracked pothole."
#             )


#             st_frame = st.empty()

#             progress_bar = st.progress(
#                 0,
#                 text="Starting..."
#             )


#             metric1, metric2, metric3, metric4 = st.columns(4)

#             current_metric = metric1.empty()

#             unique_metric = metric2.empty()

#             detection_metric = metric3.empty()

#             speed_metric = metric4.empty()


#             # =================================================
#             # UNIQUE POTHOLE TRACKS
#             # =================================================

#             # Each entry represents one physical pothole
#             unique_potholes = {}


#             next_pothole_id = 1


#             # Total frame detections
#             total_frame_detections = 0


#             # Processing counters
#             frame_number = 0
#             processed_frames = 0

#             processing_start = time.time()


#             # =================================================
#             # VIDEO LOOP
#             # =================================================

#             while True:

#                 ret, frame = cap.read()

#                 if not ret:
#                     break

#                 frame_number += 1
#                 processed_frames += 1


#                 # =================================================
#                 # RESIZE
#                 # =================================================

#                 if frame.shape[1] > max_video_width:

#                     scale = (
#                         max_video_width
#                         /
#                         frame.shape[1]
#                     )

#                     new_width = int(
#                         frame.shape[1] * scale
#                     )

#                     new_height = int(
#                         frame.shape[0] * scale
#                     )

#                     frame = cv2.resize(
#                         frame,
#                         (
#                             new_width,
#                             new_height
#                         ),
#                         interpolation=cv2.INTER_AREA
#                     )


#                 # =================================================
#                 # YOLO TRACK
#                 # =================================================

#                 results = model.track(

#                     source=frame,

#                     conf=confidence_threshold,

#                     iou=iou_threshold,

#                     imgsz=video_imgsz,

#                     persist=True,

#                     tracker="botsort.yaml",

#                     device=DEVICE,

#                     verbose=False
#                 )

#                 result = results[0]


#                 # =================================================
#                 # CURRENT FRAME DETECTIONS
#                 # =================================================

#                 current_detections = []


#                 if result.boxes is not None:

#                     for index, box in enumerate(
#                         result.boxes
#                     ):

#                         class_id = int(
#                             box.cls[0].item()
#                         )

#                         if class_id not in pothole_class_ids:
#                             continue


#                         confidence = float(
#                             box.conf[0].item()
#                         )


#                         x1, y1, x2, y2 = (
#                             box.xyxy[0]
#                             .int()
#                             .cpu()
#                             .tolist()
#                         )


#                         center = get_box_center(
#                             (
#                                 x1,
#                                 y1,
#                                 x2,
#                                 y2
#                             )
#                         )


#                         # Tracker ID
#                         track_id = None

#                         if result.boxes.id is not None:

#                             try:

#                                 track_id = int(
#                                     result
#                                     .boxes
#                                     .id[index]
#                                     .item()
#                                 )

#                             except Exception:

#                                 track_id = None


#                         current_detections.append(
#                             {
#                                 "box": (
#                                     x1,
#                                     y1,
#                                     x2,
#                                     y2
#                                 ),
#                                 "center": center,
#                                 "confidence": confidence,
#                                 "track_id": track_id
#                             }
#                         )


#                 current_frame_potholes = len(
#                     current_detections
#                 )

#                 total_frame_detections += (
#                     current_frame_potholes
#                 )


#                 # =================================================
#                 # MARK ALL EXISTING POTHOLES AS NOT SEEN
#                 # =================================================

#                 for pothole in unique_potholes.values():

#                     pothole["seen_this_frame"] = False


#                 # =================================================
#                 # MATCH DETECTIONS TO EXISTING POTHOLES
#                 # =================================================

#                 for detection in current_detections:

#                     best_match = None
#                     best_score = float("inf")


#                     # -------------------------------------------------
#                     # FIRST: MATCH USING TRACK ID
#                     # -------------------------------------------------

#                     if detection["track_id"] is not None:

#                         for pothole_id, pothole in unique_potholes.items():

#                             if (
#                                 pothole["last_track_id"]
#                                 ==
#                                 detection["track_id"]
#                             ):

#                                 distance = center_distance(
#                                     detection["center"],
#                                     pothole["center"]
#                                 )

#                                 if (
#                                     distance
#                                     <= spatial_match_distance * 2
#                                 ):

#                                     best_match = pothole_id
#                                     break


#                     # -------------------------------------------------
#                     # SECOND: SPATIAL MATCH
#                     # -------------------------------------------------

#                     if best_match is None:

#                         for pothole_id, pothole in unique_potholes.items():

#                             if pothole["seen_this_frame"]:
#                                 continue


#                             distance = center_distance(
#                                 detection["center"],
#                                 pothole["center"]
#                             )


#                             iou = calculate_iou(
#                                 detection["box"],
#                                 pothole["box"]
#                             )


#                             # Strong spatial match
#                             if (
#                                 distance
#                                 <= spatial_match_distance
#                                 or
#                                 iou >= 0.30
#                             ):

#                                 if distance < best_score:

#                                     best_score = distance
#                                     best_match = pothole_id


#                     # -------------------------------------------------
#                     # UPDATE EXISTING POTHOLE
#                     # -------------------------------------------------

#                     if best_match is not None:

#                         pothole = unique_potholes[
#                             best_match
#                         ]


#                         pothole["center"] = (
#                             detection["center"]
#                         )

#                         pothole["box"] = (
#                             detection["box"]
#                         )

#                         pothole["last_track_id"] = (
#                             detection["track_id"]
#                         )

#                         pothole["last_seen_frame"] = (
#                             frame_number
#                         )

#                         pothole["frames_seen"] += 1

#                         pothole["seen_this_frame"] = True

#                         pothole["max_confidence"] = max(
#                             pothole["max_confidence"],
#                             detection["confidence"]
#                         )


#                     # -------------------------------------------------
#                     # CREATE NEW POTHOLE
#                     # -------------------------------------------------

#                     else:

#                         new_id = next_pothole_id

#                         next_pothole_id += 1


#                         unique_potholes[
#                             new_id
#                         ] = {

#                             "center": detection["center"],

#                             "box": detection["box"],

#                             "last_track_id": detection["track_id"],

#                             "first_seen_frame": frame_number,

#                             "last_seen_frame": frame_number,

#                             "frames_seen": 1,

#                             "seen_this_frame": True,

#                             "max_confidence": detection["confidence"]
#                         }


#                 # =================================================
#                 # REMOVE OLD / STALE TRACKS
#                 # =================================================

#                 stale_ids = []

#                 for pothole_id, pothole in unique_potholes.items():

#                     missing_frames = (
#                         frame_number
#                         -
#                         pothole["last_seen_frame"]
#                     )


#                     if missing_frames > max_missing_frames:

#                         # Only remove if it was never confirmed
#                         if (
#                             pothole["frames_seen"]
#                             <
#                             min_confirm_frames
#                         ):

#                             stale_ids.append(
#                                 pothole_id
#                             )


#                 for pothole_id in stale_ids:

#                     del unique_potholes[
#                         pothole_id
#                     ]


#                 # =================================================
#                 # CONFIRMED POTHOLES
#                 # =================================================

#                 confirmed_potholes = {

#                     pothole_id: pothole

#                     for pothole_id, pothole
#                     in unique_potholes.items()

#                     if (
#                         pothole["frames_seen"]
#                         >=
#                         min_confirm_frames
#                     )
#                 }


#                 # =================================================
#                 # DRAW DETECTIONS
#                 # =================================================

#                 for detection in current_detections:

#                     x1, y1, x2, y2 = (
#                         detection["box"]
#                     )

#                     confidence = (
#                         detection["confidence"]
#                     )

#                     track_id = (
#                         detection["track_id"]
#                     )


#                     # Find corresponding unique pothole
#                     matched_pothole_id = None


#                     for pothole_id, pothole in unique_potholes.items():

#                         if pothole["seen_this_frame"]:

#                             distance = center_distance(
#                                 detection["center"],
#                                 pothole["center"]
#                             )

#                             if (
#                                 distance
#                                 <= spatial_match_distance
#                             ):

#                                 matched_pothole_id = (
#                                     pothole_id
#                                 )

#                                 break


#                     if matched_pothole_id is not None:

#                         label = (
#                             f"Pothole "
#                             f"#{matched_pothole_id} "
#                             f"{confidence:.2f}"
#                         )

#                     else:

#                         label = (
#                             f"Pothole "
#                             f"{confidence:.2f}"
#                         )


#                     # Draw bounding box

#                     cv2.rectangle(
#                         frame,
#                         (x1, y1),
#                         (x2, y2),
#                         (0, 255, 0),
#                         2
#                     )


#                     # Draw label

#                     cv2.putText(
#                         frame,
#                         label,
#                         (
#                             x1,
#                             max(
#                                 y1 - 10,
#                                 20
#                             )
#                         ),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.55,
#                         (0, 255, 0),
#                         2
#                     )


#                 # =================================================
#                 # FRAME INFORMATION
#                 # =================================================

#                 cv2.rectangle(
#                     frame,
#                     (0, 0),
#                     (470, 110),
#                     (0, 0, 0),
#                     -1
#                 )


#                 cv2.putText(
#                     frame,
#                     f"Frame: {frame_number}/{total_frames}",
#                     (10, 25),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.60,
#                     (255, 255, 255),
#                     2
#                 )


#                 cv2.putText(
#                     frame,
#                     f"Current: {current_frame_potholes}",
#                     (10, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.60,
#                     (255, 255, 255),
#                     2
#                 )


#                 cv2.putText(
#                     frame,
#                     f"Unique Potholes: {len(confirmed_potholes)}",
#                     (10, 75),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.60,
#                     (255, 255, 255),
#                     2
#                 )


#                 cv2.putText(
#                     frame,
#                     f"Confirmed after {min_confirm_frames} frames",
#                     (10, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.50,
#                     (255, 255, 255),
#                     1
#                 )


#                 # =================================================
#                 # WRITE OUTPUT FRAME
#                 # =================================================

#                 writer.write(
#                     frame
#                 )


#                 # =================================================
#                 # DISPLAY
#                 # =================================================

#                 rgb_frame = cv2.cvtColor(
#                     frame,
#                     cv2.COLOR_BGR2RGB
#                 )

#                 st_frame.image(
#                     rgb_frame,
#                     use_container_width=True
#                 )


#                 # =================================================
#                 # SPEED
#                 # =================================================

#                 elapsed = (
#                     time.time()
#                     -
#                     processing_start
#                 )

#                 if elapsed > 0:

#                     processing_fps = (
#                         processed_frames
#                         /
#                         elapsed
#                     )

#                 else:

#                     processing_fps = 0


#                 # =================================================
#                 # METRICS
#                 # =================================================

#                 current_metric.metric(
#                     "🕳️ Current Frame",
#                     current_frame_potholes
#                 )

#                 unique_metric.metric(
#                     "🎯 Unique Potholes",
#                     len(confirmed_potholes)
#                 )

#                 detection_metric.metric(
#                     "📊 Frame Detections",
#                     total_frame_detections
#                 )

#                 speed_metric.metric(
#                     "⚡ Processing FPS",
#                     f"{processing_fps:.1f}"
#                 )


#                 # =================================================
#                 # PROGRESS
#                 # =================================================

#                 progress = (
#                     frame_number
#                     /
#                     total_frames
#                 )

#                 progress = min(
#                     max(progress, 0),
#                     1
#                 )


#                 current_video_time = (
#                     frame_number
#                     /
#                     original_fps
#                 )


#                 progress_bar.progress(

#                     progress,

#                     text=(
#                         f"Processing "
#                         f"{progress * 100:.1f}% | "
#                         f"{current_video_time:.1f}s / "
#                         f"{video_duration:.1f}s"
#                     )
#                 )


#             # =================================================
#             # RELEASE
#             # =================================================

#             cap.release()
#             writer.release()


#             processing_time = (
#                 time.time()
#                 -
#                 processing_start
#             )


#             # =================================================
#             # FINAL CONFIRMED POTHOLES
#             # =================================================

#             final_confirmed_potholes = {

#                 pothole_id: pothole

#                 for pothole_id, pothole
#                 in unique_potholes.items()

#                 if (
#                     pothole["frames_seen"]
#                     >=
#                     min_confirm_frames
#                 )
#             }


#             # =================================================
#             # FINAL RESULTS
#             # =================================================

#             st.divider()

#             st.success(
#                 "✅ Complete video processing finished!"
#             )


#             r1, r2, r3, r4 = st.columns(4)


#             r1.metric(
#                 "🕳️ Unique Potholes",
#                 len(
#                     final_confirmed_potholes
#                 )
#             )


#             r2.metric(
#                 "📊 Frame Detections",
#                 total_frame_detections
#             )


#             r3.metric(
#                 "⏱️ Processing Time",
#                 f"{processing_time:.1f}s"
#             )


#             if processing_time > 0:

#                 average_fps = (
#                     processed_frames
#                     /
#                     processing_time
#                 )

#             else:

#                 average_fps = 0


#             r4.metric(
#                 "⚡ Average FPS",
#                 f"{average_fps:.1f}"
#             )


#             # =================================================
#             # EXPLANATION
#             # =================================================

#             st.info(
#                 "The Unique Potholes value counts confirmed "
#                 "tracked potholes rather than counting the same "
#                 "pothole once for every video frame. "
#                 "Frame Detections shows how many pothole bounding "
#                 "boxes were detected across the video."
#             )


#             # =================================================
#             # COMPLETE OUTPUT VIDEO
#             # =================================================

#             st.subheader(
#                 "🎬 Complete Processed Video"
#             )


#             if (
#                 output_video_path
#                 and os.path.exists(
#                     output_video_path
#                 )
#             ):

#                 with open(
#                     output_video_path,
#                     "rb"
#                 ) as video_file:

#                     video_bytes = video_file.read()


#                 st.video(
#                     video_bytes
#                 )


#                 st.download_button(

#                     label="⬇️ Download Processed Video",

#                     data=video_bytes,

#                     file_name=(
#                         "pothole_detection_output.mp4"
#                     ),

#                     mime="video/mp4"
#                 )


#         except Exception as e:

#             st.error(
#                 f"❌ Error processing video: {e}"
#             )


#         finally:

#             # =================================================
#             # CLEANUP INPUT
#             # =================================================

#             if (
#                 temp_video_path
#                 and os.path.exists(
#                     temp_video_path
#                 )
#             ):

#                 try:

#                     os.remove(
#                         temp_video_path
#                     )

#                 except Exception:
#                     pass




# import streamlit as st
# from ultralytics import YOLO
# import cv2
# from PIL import Image
# import tempfile
# import os
# import glob
# import time
# import torch


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Pothole Detection & Tracking",
#     page_icon="🚧",
#     layout="wide"
# )

# st.title("🚧 Pothole Detection & Tracking")
# st.write("AI-based pothole detection using a custom YOLO model.")


# # ============================================================
# # DEVICE
# # ============================================================

# if torch.cuda.is_available():
#     DEVICE = 0
#     DEVICE_NAME = torch.cuda.get_device_name(0)
# else:
#     DEVICE = "cpu"
#     DEVICE_NAME = "CPU"


# # ============================================================
# # SIDEBAR - MODEL SETTINGS
# # ============================================================

# st.sidebar.header("⚙️ Model Controls")

# confidence_threshold = st.sidebar.slider(
#     "Detection Confidence",
#     min_value=0.05,
#     max_value=1.0,
#     value=0.25,
#     step=0.05
# )

# iou_threshold = st.sidebar.slider(
#     "IoU Threshold",
#     min_value=0.1,
#     max_value=0.9,
#     value=0.45,
#     step=0.05
# )

# enable_tta = st.sidebar.checkbox(
#     "Enable TTA for Images",
#     value=False,
#     help="Test Time Augmentation improves detection but is slower."
# )


# # ============================================================
# # SIDEBAR - VIDEO SETTINGS
# # ============================================================

# st.sidebar.header("🎥 Video Settings")

# frame_skip = st.sidebar.selectbox(
#     "Process Every Nth Frame",
#     options=[1, 2, 3, 4],
#     index=1
# )

# video_imgsz = st.sidebar.selectbox(
#     "Video Image Size",
#     options=[320, 416, 512, 640],
#     index=2
# )

# max_video_width = st.sidebar.selectbox(
#     "Maximum Video Width",
#     options=[640, 800, 960, 1280],
#     index=2
# )


# # ============================================================
# # HARDWARE INFORMATION
# # ============================================================

# st.sidebar.header("💻 Hardware")

# if torch.cuda.is_available():
#     st.sidebar.success(
#         f"🚀 GPU: {DEVICE_NAME}"
#     )
# else:
#     st.sidebar.info(
#         "💻 CUDA GPU not detected. Using CPU."
#     )


# # ============================================================
# # LOAD CUSTOM POTHOLE MODEL
# # ============================================================

# @st.cache_resource
# def load_pothole_model():

#     # --------------------------------------------------------
#     # OPTION 1:
#     # best.pt in the same folder as app.py
#     # --------------------------------------------------------

#     root_model = "best.pt"

#     if os.path.exists(root_model):
#         return YOLO(root_model), root_model


#     # --------------------------------------------------------
#     # OPTION 2:
#     # Search inside runs/
#     # --------------------------------------------------------

#     search_patterns = [
#         "runs/**/best.pt",
#         "./runs/**/best.pt"
#     ]

#     found_weights = []

#     for pattern in search_patterns:
#         found_weights.extend(
#             glob.glob(
#                 pattern,
#                 recursive=True
#             )
#         )

#     # Remove duplicates
#     found_weights = list(
#         dict.fromkeys(found_weights)
#     )

#     if found_weights:

#         # Prefer the latest modified model
#         found_weights.sort(
#             key=os.path.getmtime,
#             reverse=True
#         )

#         model_path = found_weights[0]

#         return YOLO(model_path), model_path


#     # --------------------------------------------------------
#     # MODEL NOT FOUND
#     # --------------------------------------------------------

#     raise FileNotFoundError(
#         "best.pt was not found. "
#         "Place your trained best.pt file "
#         "in the same folder as app.py."
#     )


# # ============================================================
# # LOAD MODEL
# # ============================================================

# try:

#     model, model_path = load_pothole_model()

#     st.sidebar.success(
#         "✅ Custom Model Loaded"
#     )

#     st.sidebar.write(
#         f"**Model:** `{model_path}`"
#     )

# except Exception as e:

#     st.error(
#         f"❌ Could not load the pothole model:\n\n{e}"
#     )

#     st.info(
#         "Make sure your trained `best.pt` file "
#         "is present in the project folder."
#     )

#     st.stop()


# # ============================================================
# # SHOW MODEL CLASSES
# # ============================================================

# st.sidebar.subheader("🔍 Model Classes")

# st.sidebar.write(model.names)

# class_names = list(model.names.values())


# # ============================================================
# # CHECK FOR POTHOLE CLASS
# # ============================================================

# pothole_class_ids = []

# for class_id, class_name in model.names.items():

#     if "pothole" in str(class_name).lower():

#         pothole_class_ids.append(
#             int(class_id)
#         )


# if len(pothole_class_ids) > 0:

#     st.sidebar.success(
#         "🕳️ Pothole class detected!"
#     )

# else:

#     st.sidebar.error(
#         "❌ Pothole class NOT found!"
#     )

#     st.warning(
#         "Your `best.pt` does not appear to contain "
#         "a pothole class."
#     )

#     st.write(
#         "Model classes:"
#     )

#     st.write(model.names)

#     st.stop()


# # ============================================================
# # INPUT MODE
# # ============================================================

# mode = st.radio(
#     "Select Input Source",
#     ["Image", "Video"],
#     horizontal=True
# )


# # ============================================================
# # IMAGE DETECTION
# # ============================================================

# if mode == "Image":

#     st.header("🖼️ Pothole Detection")

#     uploaded_file = st.file_uploader(
#         "Upload a road image",
#         type=[
#             "jpg",
#             "jpeg",
#             "png"
#         ]
#     )


#     if uploaded_file is not None:

#         image = Image.open(
#             uploaded_file
#         ).convert("RGB")


#         # ----------------------------------------------------
#         # DISPLAY INPUT
#         # ----------------------------------------------------

#         col1, col2 = st.columns(2)


#         with col1:

#             st.subheader(
#                 "Original Image"
#             )

#             st.image(
#                 image,
#                 width="stretch"
#             )


#         # ----------------------------------------------------
#         # RUN YOLO
#         # ----------------------------------------------------

#         with st.spinner(
#             "🔍 Detecting potholes..."
#         ):

#             results = model.predict(

#                 source=image,

#                 conf=confidence_threshold,

#                 iou=iou_threshold,

#                 imgsz=640,

#                 augment=enable_tta,

#                 device=DEVICE,

#                 verbose=False
#             )


#         result = results[0]


#         # ----------------------------------------------------
#         # ANNOTATED IMAGE
#         # ----------------------------------------------------

#         annotated_image = result.plot()


#         with col2:

#             st.subheader(
#                 "Detected Potholes"
#             )

#             st.image(
#                 annotated_image,
#                 width="stretch"
#             )


#         # ----------------------------------------------------
#         # DETECTION COUNT
#         # ----------------------------------------------------

#         if result.boxes is not None:

#             num_detections = len(
#                 result.boxes
#             )

#         else:

#             num_detections = 0


#         st.divider()


#         metric1, metric2 = st.columns(2)


#         with metric1:

#             st.metric(
#                 "🕳️ Potholes Detected",
#                 num_detections
#             )


#         with metric2:

#             if num_detections > 0:

#                 st.success(
#                     "Potholes detected!"
#                 )

#             else:

#                 st.info(
#                     "No potholes detected."
#                 )


#         # ----------------------------------------------------
#         # DETECTION DETAILS
#         # ----------------------------------------------------

#         if num_detections > 0:

#             st.subheader(
#                 "🔎 Detection Details"
#             )


#             for i, box in enumerate(
#                 result.boxes
#             ):

#                 class_id = int(
#                     box.cls[0].item()
#                 )

#                 confidence = float(
#                     box.conf[0].item()
#                 )

#                 class_name = model.names[
#                     class_id
#                 ]


#                 st.write(
#                     f"**Detection {i + 1}:** "
#                     f"{class_name} — "
#                     f"Confidence: "
#                     f"{confidence:.2%}"
#                 )


# # ============================================================
# # VIDEO DETECTION AND TRACKING
# # ============================================================

# else:

#     st.header(
#         "🎥 Pothole Detection & Tracking"
#     )


#     uploaded_video = st.file_uploader(

#         "Upload a road video",

#         type=[
#             "mp4",
#             "avi",
#             "mov",
#             "mkv"
#         ]
#     )


#     if uploaded_video is not None:

#         temp_video_path = None


#         try:

#             # ------------------------------------------------
#             # SAVE UPLOADED VIDEO
#             # ------------------------------------------------

#             with tempfile.NamedTemporaryFile(
#                 delete=False,
#                 suffix=".mp4"
#             ) as temp_file:

#                 temp_file.write(
#                     uploaded_video.read()
#                 )

#                 temp_video_path = (
#                     temp_file.name
#                 )


#             # ------------------------------------------------
#             # OPEN VIDEO
#             # ------------------------------------------------

#             cap = cv2.VideoCapture(
#                 temp_video_path
#             )


#             if not cap.isOpened():

#                 st.error(
#                     "❌ Could not open video."
#                 )

#                 st.stop()


#             # ------------------------------------------------
#             # VIDEO INFORMATION
#             # ------------------------------------------------

#             total_frames = int(
#                 cap.get(
#                     cv2.CAP_PROP_FRAME_COUNT
#                 )
#             )


#             original_fps = cap.get(
#                 cv2.CAP_PROP_FPS
#             )


#             if original_fps <= 0:

#                 original_fps = 30.0


#             video_width = int(
#                 cap.get(
#                     cv2.CAP_PROP_FRAME_WIDTH
#                 )
#             )


#             video_height = int(
#                 cap.get(
#                     cv2.CAP_PROP_FRAME_HEIGHT
#                 )
#             )


#             if total_frames > 0:

#                 video_duration = (
#                     total_frames /
#                     original_fps
#                 )

#             else:

#                 video_duration = 0


#             # ------------------------------------------------
#             # VIDEO INFORMATION DISPLAY
#             # ------------------------------------------------

#             info1, info2, info3, info4 = (
#                 st.columns(4)
#             )


#             info1.metric(
#                 "Duration",
#                 f"{video_duration:.1f}s"
#             )


#             info2.metric(
#                 "FPS",
#                 f"{original_fps:.1f}"
#             )


#             info3.metric(
#                 "Resolution",
#                 f"{video_width} × {video_height}"
#             )


#             info4.metric(
#                 "Frames",
#                 total_frames
#             )


#             st.divider()


#             st.subheader(
#                 "🚀 Processing Video"
#             )


#             st.write(
#                 f"Running on: **{DEVICE_NAME}**"
#             )


#             if DEVICE == "cpu":

#                 st.caption(
#                     "CPU mode detected. "
#                     "Frame skipping and resized "
#                     "frames are being used."
#                 )


#             # ------------------------------------------------
#             # STREAMING FRAME
#             # ------------------------------------------------

#             video_display = st.empty()


#             # ------------------------------------------------
#             # PROGRESS BAR
#             # ------------------------------------------------

#             progress_bar = st.progress(
#                 0,
#                 text="Starting video processing..."
#             )


#             # ------------------------------------------------
#             # LIVE METRICS
#             # ------------------------------------------------

#             metric1, metric2, metric3 = (
#                 st.columns(3)
#             )


#             current_frame_metric = (
#                 metric1.empty()
#             )

#             track_metric = (
#                 metric2.empty()
#             )

#             speed_metric = (
#                 metric3.empty()
#             )


#             # ------------------------------------------------
#             # TRACKING VARIABLES
#             # ------------------------------------------------

#             unique_track_ids = set()

#             frame_number = 0

#             processed_frames = 0

#             processing_start = time.time()


#             # ------------------------------------------------
#             # VIDEO LOOP
#             # ------------------------------------------------

#             while cap.isOpened():

#                 ret, frame = cap.read()


#                 if not ret:

#                     break


#                 frame_number += 1


#                 # --------------------------------------------
#                 # FRAME SKIPPING
#                 # --------------------------------------------

#                 if (
#                     frame_number %
#                     frame_skip != 0
#                 ):

#                     continue


#                 processed_frames += 1


#                 # --------------------------------------------
#                 # RESIZE FRAME
#                 # --------------------------------------------

#                 height, width = (
#                     frame.shape[:2]
#                 )


#                 if width > max_video_width:

#                     scale = (
#                         max_video_width /
#                         width
#                     )


#                     new_width = int(
#                         width * scale
#                     )


#                     new_height = int(
#                         height * scale
#                     )


#                     frame = cv2.resize(

#                         frame,

#                         (
#                             new_width,
#                             new_height
#                         ),

#                         interpolation=cv2.INTER_AREA
#                     )


#                 # --------------------------------------------
#                 # YOLO TRACKING
#                 # --------------------------------------------

#                 results = model.track(

#                     source=frame,

#                     conf=confidence_threshold,

#                     iou=iou_threshold,

#                     imgsz=video_imgsz,

#                     persist=True,

#                     tracker="botsort.yaml",

#                     device=DEVICE,

#                     verbose=False
#                 )


#                 result = results[0]


#                 # --------------------------------------------
#                 # CURRENT FRAME DETECTIONS
#                 # --------------------------------------------

#                 frame_potholes = 0


#                 if (
#                     result.boxes is not None
#                     and
#                     len(result.boxes) > 0
#                 ):


#                     # ----------------------------------------
#                     # FILTER ONLY POTHOLES
#                     # ----------------------------------------

#                     pothole_boxes = []


#                     for box in result.boxes:

#                         class_id = int(
#                             box.cls[0].item()
#                         )


#                         if (
#                             class_id
#                             in pothole_class_ids
#                         ):

#                             pothole_boxes.append(
#                                 box
#                             )


#                     frame_potholes = len(
#                         pothole_boxes
#                     )


#                     # ----------------------------------------
#                     # TRACK IDS
#                     # ----------------------------------------

#                     if (
#                         result.boxes.id
#                         is not None
#                     ):

#                         track_ids = (
#                             result.boxes.id
#                             .int()
#                             .cpu()
#                             .tolist()
#                         )


#                         for track_id in track_ids:

#                             unique_track_ids.add(
#                                 track_id
#                             )


#                 # --------------------------------------------
#                 # DRAW RESULTS
#                 # --------------------------------------------

#                 annotated_frame = (
#                     result.plot()
#                 )


#                 # --------------------------------------------
#                 # CONVERT BGR → RGB
#                 # --------------------------------------------

#                 rgb_frame = cv2.cvtColor(

#                     annotated_frame,

#                     cv2.COLOR_BGR2RGB
#                 )


#                 # --------------------------------------------
#                 # DISPLAY FRAME
#                 # --------------------------------------------

#                 video_display.image(

#                     rgb_frame,

#                     width="stretch"
#                 )


#                 # --------------------------------------------
#                 # UPDATE METRICS
#                 # --------------------------------------------

#                 current_frame_metric.metric(

#                     "🕳️ Potholes in Frame",

#                     frame_potholes
#                 )


#                 track_metric.metric(

#                     "🎯 Track IDs",

#                     len(unique_track_ids)
#                 )


#                 # --------------------------------------------
#                 # PROCESSING SPEED
#                 # --------------------------------------------

#                 elapsed = (
#                     time.time() -
#                     processing_start
#                 )


#                 if elapsed > 0:

#                     processing_fps = (
#                         processed_frames /
#                         elapsed
#                     )

#                 else:

#                     processing_fps = 0


#                 speed_metric.metric(

#                     "⚡ Processing FPS",

#                     f"{processing_fps:.1f}"
#                 )


#                 # --------------------------------------------
#                 # PROGRESS
#                 # --------------------------------------------

#                 if total_frames > 0:

#                     progress = min(

#                         frame_number /
#                         total_frames,

#                         1.0
#                     )

#                 else:

#                     progress = 0


#                 processed_video_time = (

#                     frame_number /
#                     original_fps
#                 )


#                 progress_bar.progress(

#                     progress,

#                     text=(

#                         f"Processing "
#                         f"{progress * 100:.1f}% | "

#                         f"Video: "
#                         f"{processed_video_time:.1f}s / "
#                         f"{video_duration:.1f}s"
#                     )
#                 )


#             # ------------------------------------------------
#             # RELEASE VIDEO
#             # ------------------------------------------------

#             cap.release()


#             processing_time = (
#                 time.time() -
#                 processing_start
#             )


#             # ------------------------------------------------
#             # FINAL RESULTS
#             # ------------------------------------------------

#             st.divider()


#             st.success(
#                 "✅ Video processing completed!"
#             )


#             result1, result2, result3 = (
#                 st.columns(3)
#             )


#             result1.metric(

#                 "🎯 Unique Track IDs",

#                 len(unique_track_ids)
#             )


#             result2.metric(

#                 "⏱️ Processing Time",

#                 f"{processing_time:.1f}s"
#             )


#             if processing_time > 0:

#                 average_fps = (

#                     processed_frames /
#                     processing_time
#                 )

#             else:

#                 average_fps = 0


#             result3.metric(

#                 "⚡ Average FPS",

#                 f"{average_fps:.1f}"
#             )


#             # ------------------------------------------------
#             # SPEED COMPARISON
#             # ------------------------------------------------

#             if video_duration > 0:

#                 if (
#                     processing_time <
#                     video_duration
#                 ):

#                     realtime_multiplier = (

#                         video_duration /
#                         processing_time
#                     )


#                     st.info(

#                         f"Processing speed: "
#                         f"{realtime_multiplier:.2f}× "
#                         f"the video duration."
#                     )

#                 else:

#                     slowdown = (

#                         processing_time /
#                         video_duration
#                     )


#                     st.info(

#                         f"Processing took "
#                         f"{slowdown:.2f}× "
#                         f"the video duration."
#                     )


#         except Exception as e:

#             st.error(
#                 f"❌ Error processing video: {e}"
#             )


#         finally:

#             if (
#                 temp_video_path
#                 is not None
#                 and
#                 os.path.exists(
#                     temp_video_path
#                 )
#             ):

#                 try:

#                     os.remove(
#                         temp_video_path
#                     )

#                 except Exception:

#                     pass
