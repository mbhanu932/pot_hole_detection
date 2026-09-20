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

st.write(
    "Detect, track and count potholes throughout an entire road video "
    "using a fine-tuned YOLO model."
)


# ============================================================
# DEVICE DETECTION
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)
else:
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model Settings")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    0.05,
    1.0,
    0.25,
    0.05
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    0.10,
    0.90,
    0.45,
    0.05
)

enable_tta = st.sidebar.checkbox(
    "Enable TTA for Images",
    value=False
)


# ============================================================
# VIDEO SETTINGS
# ============================================================

st.sidebar.header("🎥 Video Settings")

video_imgsz = st.sidebar.selectbox(
    "Video Inference Size",
    [320, 416, 512, 640],
    index=2
)

max_video_width = st.sidebar.selectbox(
    "Maximum Video Width",
    [640, 800, 960, 1280],
    index=2
)

st.sidebar.info(
    "Every frame of the video will be processed."
)


# ============================================================
# HARDWARE INFORMATION
# ============================================================

st.sidebar.header("💻 Hardware")

if torch.cuda.is_available():

    st.sidebar.success(
        f"🚀 GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    st.sidebar.info(
        "💻 CUDA GPU not detected. Using CPU."
    )


# ============================================================
# LOAD CUSTOM MODEL
# ============================================================

@st.cache_resource
def load_pothole_model():

    # --------------------------------------------------------
    # FIRST: best.pt in application directory
    # --------------------------------------------------------

    if os.path.exists("best.pt"):
        return YOLO("best.pt")

    # --------------------------------------------------------
    # SECOND: search inside runs folder
    # --------------------------------------------------------

    found_weights = glob.glob(
        "runs/**/best.pt",
        recursive=True
    )

    if found_weights:

        found_weights.sort(
            key=os.path.getmtime,
            reverse=True
        )

        return YOLO(found_weights[0])

    # --------------------------------------------------------
    # DO NOT USE BASE YOLO MODEL
    # --------------------------------------------------------

    raise FileNotFoundError(
        "Custom best.pt model was not found. "
        "Please place best.pt in the same folder as app.py."
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_pothole_model()

    st.sidebar.success(
        "✅ Custom Pothole Model Loaded"
    )

    st.sidebar.write(
        "**Model Classes:**",
        model.names
    )

except Exception as e:

    st.error(
        f"❌ Could not load model: {e}"
    )

    st.stop()


# ============================================================
# FIND POTHOLE CLASS
# ============================================================

pothole_class_ids = set()

for class_id, class_name in model.names.items():

    if str(class_name).lower() == "pothole":

        pothole_class_ids.add(
            int(class_id)
        )


if not pothole_class_ids:

    st.error(
        "❌ The loaded model does not contain a 'pothole' class."
    )

    st.write(
        "Available classes:",
        model.names
    )

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

    st.header("🖼️ Pothole Detection on Image")

    uploaded_file = st.file_uploader(
        "Upload Road Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "Original Image"
            )

            st.image(
                image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "🔍 Detected Potholes"
            )

            with st.spinner(
                "Detecting potholes..."
            ):

                results = model.predict(
                    source=image,
                    conf=confidence_threshold,
                    iou=iou_threshold,
                    imgsz=1280,
                    augment=enable_tta,
                    device=DEVICE,
                    verbose=False
                )

            result = results[0]

            # ------------------------------------------------
            # ONLY POTHOLE DETECTIONS
            # ------------------------------------------------

            pothole_count = 0

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    if class_id in pothole_class_ids:
                        pothole_count += 1

            # ------------------------------------------------
            # PLOT RESULT
            # ------------------------------------------------

            annotated_image = result.plot()

            st.image(
                annotated_image,
                caption="Pothole Detection Result",
                use_container_width=True
            )

            st.metric(
                "🕳️ Potholes Detected",
                pothole_count
            )


# ============================================================
# VIDEO DETECTION
# ============================================================

elif mode == "Video":

    st.header(
        "🎥 Full Video Pothole Detection & Tracking"
    )

    uploaded_video = st.file_uploader(
        "Upload Road Video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_video is not None:

        temp_video_path = None
        output_video_path = None

        try:

            # =================================================
            # SAVE UPLOADED VIDEO
            # =================================================

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(
                    uploaded_video.read()
                )

                temp_video_path = temp_file.name


            # =================================================
            # OPEN VIDEO
            # =================================================

            cap = cv2.VideoCapture(
                temp_video_path
            )

            if not cap.isOpened():

                st.error(
                    "❌ Could not open uploaded video."
                )

                st.stop()


            # =================================================
            # VIDEO INFORMATION
            # =================================================

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

            original_width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            original_height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            video_duration = (
                total_frames / original_fps
            )


            # =================================================
            # VIDEO INFORMATION
            # =================================================

            info1, info2, info3, info4 = st.columns(4)

            info1.metric(
                "Video Duration",
                f"{video_duration:.1f}s"
            )

            info2.metric(
                "Original FPS",
                f"{original_fps:.1f}"
            )

            info3.metric(
                "Resolution",
                f"{original_width} × {original_height}"
            )

            info4.metric(
                "Total Frames",
                total_frames
            )


            st.divider()


            # =================================================
            # DETERMINE PROCESSING SIZE
            # =================================================

            if original_width > max_video_width:

                scale = (
                    max_video_width
                    / original_width
                )

                output_width = int(
                    original_width * scale
                )

                output_height = int(
                    original_height * scale
                )

            else:

                output_width = original_width
                output_height = original_height


            # =================================================
            # CREATE OUTPUT VIDEO
            # =================================================

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_video_path = output_file.name

            output_file.close()


            # Try MP4V codec
            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                original_fps,
                (
                    output_width,
                    output_height
                )
            )

            if not writer.isOpened():

                raise RuntimeError(
                    "Could not create output video."
                )


            # =================================================
            # PROCESSING UI
            # =================================================

            st.subheader(
                "🚀 Processing Entire Video"
            )

            st.write(
                f"Running inference on: **{DEVICE_NAME}**"
            )

            st.info(
                "Every frame is being processed from "
                "the beginning to the end of the video."
            )


            # =================================================
            # LIVE VIDEO DISPLAY
            # =================================================

            st_frame = st.empty()


            # =================================================
            # PROGRESS BAR
            # =================================================

            progress_bar = st.progress(
                0,
                text="Starting video processing..."
            )


            # =================================================
            # LIVE METRICS
            # =================================================

            metric1, metric2, metric3, metric4 = st.columns(4)

            current_metric = metric1.empty()

            confirmed_metric = metric2.empty()

            detection_metric = metric3.empty()

            speed_metric = metric4.empty()


            # =================================================
            # TRACKING VARIABLES
            # =================================================

            # Number of frames in which each track appeared
            track_frame_count = {}

            # Confirmed tracker IDs
            confirmed_track_ids = set()

            # Tracks visible in current frame
            current_track_ids = set()

            # Minimum frames required to confirm a track
            MIN_CONFIRM_FRAMES = 5

            # Total bounding-box detections
            total_pothole_detections = 0

            # Frame counter
            frame_number = 0

            # Processed frame counter
            processed_frames = 0

            # Start timer
            processing_start = time.time()


            # =================================================
            # VIDEO LOOP
            # =================================================

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1


                # =================================================
                # RESIZE FRAME
                # =================================================

                if frame.shape[1] > max_video_width:

                    scale = (
                        max_video_width
                        / frame.shape[1]
                    )

                    new_width = int(
                        frame.shape[1] * scale
                    )

                    new_height = int(
                        frame.shape[0] * scale
                    )

                    frame = cv2.resize(
                        frame,
                        (
                            new_width,
                            new_height
                        ),
                        interpolation=cv2.INTER_AREA
                    )


                processed_frames += 1


                # =================================================
                # YOLO TRACKING
                # =================================================

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


                # =================================================
                # RESET CURRENT FRAME TRACKS
                # =================================================

                current_track_ids = set()

                current_frame_potholes = 0


                # =================================================
                # PROCESS DETECTIONS
                # =================================================

                if result.boxes is not None:

                    for index, box in enumerate(
                        result.boxes
                    ):

                        # -----------------------------------------
                        # CLASS
                        # -----------------------------------------

                        class_id = int(
                            box.cls[0].item()
                        )


                        # -----------------------------------------
                        # ONLY POTHOLE CLASS
                        # -----------------------------------------

                        if class_id not in pothole_class_ids:
                            continue


                        # -----------------------------------------
                        # CONFIDENCE
                        # -----------------------------------------

                        confidence = float(
                            box.conf[0].item()
                        )


                        # -----------------------------------------
                        # COUNT DETECTION
                        # -----------------------------------------

                        current_frame_potholes += 1

                        total_pothole_detections += 1


                        # -----------------------------------------
                        # GET TRACK ID
                        # -----------------------------------------

                        track_id = None

                        if result.boxes.id is not None:

                            try:

                                track_id = int(
                                    result
                                    .boxes
                                    .id[index]
                                    .item()
                                )

                            except Exception:

                                track_id = None


                        # -----------------------------------------
                        # TRACK ID PROCESSING
                        # -----------------------------------------

                        if track_id is not None:

                            current_track_ids.add(
                                track_id
                            )


                            # Create counter
                            if track_id not in track_frame_count:

                                track_frame_count[
                                    track_id
                                ] = 0


                            # Increment appearance count
                            track_frame_count[
                                track_id
                            ] += 1


                            # Confirm track
                            if (
                                track_frame_count[
                                    track_id
                                ]
                                >= MIN_CONFIRM_FRAMES
                            ):

                                confirmed_track_ids.add(
                                    track_id
                                )


                        # -----------------------------------------
                        # DRAW BOX
                        # -----------------------------------------

                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .int()
                            .cpu()
                            .tolist()
                        )


                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )


                        # -----------------------------------------
                        # LABEL
                        # -----------------------------------------

                        if track_id is not None:

                            label = (
                                f"Pothole "
                                f"ID:{track_id} "
                                f"{confidence:.2f}"
                            )

                        else:

                            label = (
                                f"Pothole "
                                f"{confidence:.2f}"
                            )


                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2
                        )


                # =================================================
                # ADD INFORMATION TO FRAME
                # =================================================

                cv2.rectangle(
                    frame,
                    (0, 0),
                    (430, 100),
                    (0, 0, 0),
                    -1
                )


                cv2.putText(
                    frame,
                    f"Frame: {frame_number}/{total_frames}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Current Potholes: {current_frame_potholes}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )


                cv2.putText(
                    frame,
                    f"Confirmed Tracks: {len(confirmed_track_ids)}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )


                # =================================================
                # WRITE FRAME TO OUTPUT VIDEO
                # =================================================

                writer.write(
                    frame
                )


                # =================================================
                # DISPLAY CURRENT FRAME
                # =================================================

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                st_frame.image(
                    rgb_frame,
                    use_container_width=True
                )


                # =================================================
                # PROCESSING SPEED
                # =================================================

                elapsed = (
                    time.time()
                    - processing_start
                )

                if elapsed > 0:

                    processing_fps = (
                        processed_frames
                        / elapsed
                    )

                else:

                    processing_fps = 0


                # =================================================
                # UPDATE METRICS
                # =================================================

                current_metric.metric(
                    "🕳️ Current Frame",
                    current_frame_potholes
                )

                confirmed_metric.metric(
                    "🎯 Confirmed Track IDs",
                    len(confirmed_track_ids)
                )

                detection_metric.metric(
                    "📊 Total Detections",
                    total_pothole_detections
                )

                speed_metric.metric(
                    "⚡ Processing FPS",
                    f"{processing_fps:.1f}"
                )


                # =================================================
                # PROGRESS
                # =================================================

                progress = (
                    frame_number
                    / total_frames
                )

                progress = min(
                    max(progress, 0.0),
                    1.0
                )


                video_time = (
                    frame_number
                    / original_fps
                )


                progress_bar.progress(

                    progress,

                    text=(
                        f"Processing "
                        f"{progress * 100:.1f}% | "
                        f"{video_time:.1f}s / "
                        f"{video_duration:.1f}s"
                    )
                )


            # =====================================================
            # RELEASE VIDEO
            # =====================================================

            cap.release()

            writer.release()


            # =====================================================
            # PROCESSING TIME
            # =====================================================

            processing_time = (
                time.time()
                - processing_start
            )


            # =====================================================
            # FINAL RESULTS
            # =====================================================

            st.divider()

            st.success(
                "✅ Complete video processing finished!"
            )


            result1, result2, result3, result4 = st.columns(4)


            result1.metric(
                "🎯 Confirmed Track IDs",
                len(confirmed_track_ids)
            )


            result2.metric(
                "📊 Total Pothole Detections",
                total_pothole_detections
            )


            result3.metric(
                "⏱️ Processing Time",
                f"{processing_time:.1f}s"
            )


            if processing_time > 0:

                average_fps = (
                    processed_frames
                    / processing_time
                )

            else:

                average_fps = 0


            result4.metric(
                "⚡ Average FPS",
                f"{average_fps:.1f}"
            )


            # =================================================
            # EXPLANATION
            # =================================================

            st.info(
                "ℹ️ Total Pothole Detections = every pothole "
                "bounding box detected across all processed frames. "
                "Confirmed Track IDs = tracker identities that "
                "persisted for at least 5 frames. A Track ID is "
                "not guaranteed to represent one physical pothole."
            )


            # =================================================
            # SHOW COMPLETE PROCESSED VIDEO
            # =================================================

            st.subheader(
                "🎬 Complete Processed Video"
            )


            if (
                output_video_path is not None
                and os.path.exists(output_video_path)
            ):

                with open(
                    output_video_path,
                    "rb"
                ) as video_file:

                    video_bytes = video_file.read()


                st.video(
                    video_bytes
                )


                # =================================================
                # DOWNLOAD BUTTON
                # =================================================

                st.download_button(

                    label="⬇️ Download Processed Video",

                    data=video_bytes,

                    file_name=(
                        "pothole_detection_output.mp4"
                    ),

                    mime="video/mp4"
                )


        except Exception as e:

            st.error(
                f"❌ Error processing video: {e}"
            )


        finally:

            # =====================================================
            # CLEANUP INPUT VIDEO
            # =====================================================

            if (
                temp_video_path is not None
                and os.path.exists(
                    temp_video_path
                )
            ):

                try:

                    os.remove(
                        temp_video_path
                    )

                except Exception:
                    pass




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
