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
    "Detect and track potholes in road images and videos "
    "using a fine-tuned YOLO model."
)


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
# SIDEBAR
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
    value=False
)


# ============================================================
# VIDEO SETTINGS
# ============================================================

st.sidebar.header("🎥 Video Settings")

frame_skip = st.selectbox(
    "Process Every Nth Frame",
    options=[1, 2, 3],
    index=0,
    help=(
        "1 = process every frame. "
        "2 = process every second frame. "
        "3 = process every third frame."
    )
)

video_imgsz = st.selectbox(
    "Video Inference Size",
    options=[320, 416, 512, 640],
    index=2
)

max_video_width = st.selectbox(
    "Maximum Video Width",
    options=[640, 800, 960, 1280],
    index=2
)


# ============================================================
# HARDWARE
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
# LOAD CUSTOM MODEL
# ============================================================

@st.cache_resource
def load_pothole_model():

    # First look for best.pt beside app.py
    root_model = "best.pt"

    if os.path.exists(root_model):

        return YOLO(root_model), root_model


    # If not found, search inside runs/
    found_weights = glob.glob(
        "runs/**/best.pt",
        recursive=True
    )

    if found_weights:

        found_weights.sort(
            key=os.path.getmtime,
            reverse=True
        )

        model_path = found_weights[0]

        return YOLO(model_path), model_path


    raise FileNotFoundError(
        "best.pt was not found. "
        "Place best.pt in the same folder as app.py."
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, model_path = load_pothole_model()

    st.sidebar.success(
        "✅ Custom Pothole Model Loaded"
    )

    st.sidebar.write(
        f"Model: `{model_path}`"
    )

except Exception as e:

    st.error(
        f"❌ Could not load model: {e}"
    )

    st.info(
        "Make sure your valid best.pt file is "
        "in the project folder."
    )

    st.stop()


# ============================================================
# MODEL CLASSES
# ============================================================

st.sidebar.subheader("🔍 Model Classes")

st.sidebar.write(model.names)


# ============================================================
# FIND POTHOLE CLASS
# ============================================================

pothole_class_ids = []

for class_id, class_name in model.names.items():

    if "pothole" in str(class_name).lower():

        pothole_class_ids.append(
            int(class_id)
        )


if not pothole_class_ids:

    st.sidebar.error(
        "❌ Pothole class not found!"
    )

    st.error(
        "Your model does not contain a class named 'pothole'."
    )

    st.stop()


st.sidebar.success(
    "🕳️ Pothole class detected!"
)


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
                width="stretch"
            )


        # ----------------------------------------------------
        # DETECTION
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
        # FILTER POTHOLES
        # ----------------------------------------------------

        pothole_count = 0

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                if class_id in pothole_class_ids:

                    pothole_count += 1


        # ----------------------------------------------------
        # DISPLAY RESULT
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


        st.divider()


        metric1, metric2 = st.columns(2)


        with metric1:

            st.metric(
                "🕳️ Potholes Detected",
                pothole_count
            )


        with metric2:

            if pothole_count > 0:

                st.success(
                    "Potholes detected!"
                )

            else:

                st.info(
                    "No potholes detected."
                )


        # ----------------------------------------------------
        # DETAILS
        # ----------------------------------------------------

        if pothole_count > 0:

            st.subheader(
                "🔎 Detection Details"
            )


            detection_number = 0


            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )


                if class_id not in pothole_class_ids:

                    continue


                detection_number += 1


                confidence = float(
                    box.conf[0].item()
                )


                st.write(
                    f"**Pothole {detection_number}** "
                    f"— Confidence: "
                    f"{confidence:.2%}"
                )


# ============================================================
# VIDEO DETECTION + TRACKING
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

        input_video_path = None
        output_video_path = None


        try:

            # ==================================================
            # SAVE INPUT VIDEO
            # ==================================================

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(
                    uploaded_video.read()
                )

                input_video_path = (
                    temp_file.name
                )


            # ==================================================
            # OPEN VIDEO
            # ==================================================

            cap = cv2.VideoCapture(
                input_video_path
            )


            if not cap.isOpened():

                st.error(
                    "❌ Could not open the uploaded video."
                )

                st.stop()


            # ==================================================
            # VIDEO INFORMATION
            # ==================================================

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


            if total_frames > 0:

                video_duration = (
                    total_frames /
                    original_fps
                )

            else:

                video_duration = 0


            # ==================================================
            # DISPLAY VIDEO INFORMATION
            # ==================================================

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
                f"{original_width} × {original_height}"
            )


            info4.metric(
                "Frames",
                total_frames
            )


            st.divider()


            # ==================================================
            # PROCESSING MESSAGE
            # ==================================================

            st.subheader(
                "🚀 Processing Complete Video"
            )


            st.write(
                f"Running on: **{DEVICE_NAME}**"
            )


            if frame_skip == 1:

                st.info(
                    "Every video frame will be processed "
                    "for smoother tracking."
                )

            else:

                st.warning(
                    f"Every {frame_skip}th frame is being "
                    f"processed."
                )


            # ==================================================
            # OUTPUT VIDEO SIZE
            # ==================================================

            if original_width > max_video_width:

                scale = (
                    max_video_width /
                    original_width
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


            # Ensure dimensions are even
            output_width -= output_width % 2
            output_height -= output_height % 2


            # ==================================================
            # OUTPUT FILE
            # ==================================================

            output_video_path = (
                tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name
            )


            # ==================================================
            # VIDEO WRITER
            # ==================================================

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


            # ==================================================
            # TRACKING VARIABLES
            # ==================================================

            confirmed_track_ids = set()

            track_frame_count = {}

            missed_frames = {}

            MIN_CONFIRM_FRAMES = 3

            MAX_MISSED_FRAMES = 30


            frame_number = 0

            processed_frames = 0

            total_pothole_detections = 0

            processing_start = time.time()


            # ==================================================
            # PROGRESS UI
            # ==================================================

            progress_bar = st.progress(
                0,
                text="Starting..."
            )


            status_text = st.empty()


            metric1, metric2, metric3 = (
                st.columns(3)
            )


            current_metric = metric1.empty()

            confirmed_metric = metric2.empty()

            speed_metric = metric3.empty()


            # ==================================================
            # VIDEO PROCESSING LOOP
            # ==================================================

            while True:

                ret, frame = cap.read()


                if not ret:

                    break


                frame_number += 1


                # ------------------------------------------------
                # PROCESS EVERY NTH FRAME
                # ------------------------------------------------

                should_process = (
                    frame_number % frame_skip == 0
                )


                # ------------------------------------------------
                # If frame is skipped, still write it
                # to maintain original video length.
                # ------------------------------------------------

                if not should_process:

                    resized_frame = frame


                    if (
                        original_width >
                        max_video_width
                    ):

                        resized_frame = cv2.resize(

                            frame,

                            (
                                output_width,
                                output_height
                            ),

                            interpolation=cv2.INTER_AREA
                        )


                    writer.write(
                        resized_frame
                    )

                    continue


                processed_frames += 1


                # ==================================================
                # RESIZE FRAME
                # ==================================================

                if original_width > max_video_width:

                    frame = cv2.resize(

                        frame,

                        (
                            output_width,
                            output_height
                        ),

                        interpolation=cv2.INTER_AREA
                    )


                # ==================================================
                # YOLO TRACKING
                # ==================================================

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


                # ==================================================
                # CREATE POTHOLE-ONLY DISPLAY
                # ==================================================

                annotated_frame = frame.copy()


                current_potholes = 0

                current_track_ids = set()


                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0].item()
                        )


                        # Only potholes
                        if (
                            class_id
                            not in pothole_class_ids
                        ):

                            continue


                        confidence = float(
                            box.conf[0].item()
                        )


                        # Bounding box
                        x1, y1, x2, y2 = (
                            box.xyxy[0]
                            .int()
                            .cpu()
                            .tolist()
                        )


                        # Track ID
                        track_id = None


                        if (
                            result.boxes.id
                            is not None
                        ):

                            track_id = int(
                                box.id[0].item()
                            )


                        current_potholes += 1

                        total_pothole_detections += 1


                        if track_id is not None:

                            current_track_ids.add(
                                track_id
                            )


                            # ----------------------------------
                            # Track frame count
                            # ----------------------------------

                            if track_id not in track_frame_count:

                                track_frame_count[
                                    track_id
                                ] = 0


                            track_frame_count[
                                track_id
                            ] += 1


                            missed_frames[
                                track_id
                            ] = 0


                            # ----------------------------------
                            # Confirm after 3 frames
                            # ----------------------------------

                            if (
                                track_frame_count[
                                    track_id
                                ]
                                >= MIN_CONFIRM_FRAMES
                            ):

                                confirmed_track_ids.add(
                                    track_id
                                )


                        # ==================================================
                        # DRAW BOUNDING BOX
                        # ==================================================

                        cv2.rectangle(

                            annotated_frame,

                            (x1, y1),

                            (x2, y2),

                            (0, 255, 0),

                            3
                        )


                        # ==================================================
                        # LABEL
                        # ==================================================

                        if track_id is not None:

                            label = (
                                f"Pothole | "
                                f"ID {track_id} | "
                                f"{confidence:.0%}"
                            )

                        else:

                            label = (
                                f"Pothole | "
                                f"{confidence:.0%}"
                            )


                        # ==================================================
                        # LABEL BACKGROUND
                        # ==================================================

                        (text_width, text_height), _ = (
                            cv2.getTextSize(

                                label,

                                cv2.FONT_HERSHEY_SIMPLEX,

                                0.6,

                                2
                            )
                        )


                        cv2.rectangle(

                            annotated_frame,

                            (
                                x1,
                                max(
                                    0,
                                    y1 - text_height - 12
                                )
                            ),

                            (
                                x1 + text_width + 10,
                                y1
                            ),

                            (0, 255, 0),

                            -1
                        )


                        cv2.putText(

                            annotated_frame,

                            label,

                            (
                                x1 + 5,
                                y1 - 6
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.6,

                            (0, 0, 0),

                            2
                        )


                # ==================================================
                # UPDATE MISSED TRACKS
                # ==================================================

                for track_id in list(
                    track_frame_count.keys()
                ):

                    if (
                        track_id
                        not in current_track_ids
                    ):

                        missed_frames[
                            track_id
                        ] = (
                            missed_frames.get(
                                track_id,
                                0
                            ) + 1
                        )


                # ==================================================
                # WRITE PROCESSED FRAME
                # ==================================================

                writer.write(
                    annotated_frame
                )


                # ==================================================
                # UPDATE UI
                # ==================================================

                current_metric.metric(

                    "🕳️ Potholes in Frame",

                    current_potholes
                )


                confirmed_metric.metric(

                    "🎯 Confirmed Track IDs",

                    len(confirmed_track_ids)
                )


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


                # ==================================================
                # PROGRESS
                # ==================================================

                if total_frames > 0:

                    progress = min(

                        frame_number /
                        total_frames,

                        1.0
                    )

                else:

                    progress = 0


                progress_bar.progress(

                    progress,

                    text=(

                        f"Processing "
                        f"{progress * 100:.1f}% | "
                        f"Frame "
                        f"{frame_number}/"
                        f"{total_frames}"
                    )
                )


                status_text.write(

                    f"Processed "
                    f"{processed_frames} frames | "
                    f"Current potholes: "
                    f"{current_potholes} | "
                    f"Confirmed tracks: "
                    f"{len(confirmed_track_ids)}"
                )


            # ==================================================
            # FINISH VIDEO
            # ==================================================

            cap.release()

            writer.release()


            processing_time = (
                time.time() -
                processing_start
            )


            progress_bar.progress(
                1.0,
                text="Processing completed!"
            )


            # ==================================================
            # RESULTS
            # ==================================================

            st.divider()

            st.success(
                "✅ Complete video processing finished!"
            )


            result1, result2, result3, result4 = (
                st.columns(4)
            )


            result1.metric(

                "🎯 Confirmed Tracks",

                len(confirmed_track_ids)
            )


            result2.metric(

                "🕳️ Total Detections",

                total_pothole_detections
            )


            result3.metric(

                "⏱️ Processing Time",

                f"{processing_time:.1f}s"
            )


            result4.metric(

                "⚡ Average FPS",

                (
                    f"{processed_frames / processing_time:.1f}"
                    if processing_time > 0
                    else "0"
                )
            )


            # ==================================================
            # DISPLAY COMPLETE OUTPUT VIDEO
            # ==================================================

            st.subheader(
                "🎬 Processed Video"
            )


            if (
                output_video_path
                and
                os.path.exists(
                    output_video_path
                )
            ):

                with open(
                    output_video_path,
                    "rb"
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )


                st.video(
                    video_bytes
                )


                # ==================================================
                # DOWNLOAD BUTTON
                # ==================================================

                st.download_button(

                    label=(
                        "⬇️ Download Processed Video"
                    ),

                    data=video_bytes,

                    file_name=(
                        "pothole_detection_output.mp4"
                    ),

                    mime="video/mp4"
                )


            else:

                st.error(
                    "❌ Output video could not be created."
                )


        except Exception as e:

            st.error(
                f"❌ Error processing video: {e}"
            )

            st.exception(e)


        finally:

            # ----------------------------------------------------
            # CLEAN INPUT VIDEO
            # ----------------------------------------------------

            if (
                input_video_path
                and
                os.path.exists(
                    input_video_path
                )
            ):

                try:

                    os.remove(
                        input_video_path
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
