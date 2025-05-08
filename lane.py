
import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import time

# ========== Lane Detection Functions ==========

## added by hardik 
eta_placeholder = st.empty()
progress_bar = st.progress(0)

def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def hough_transform(image):
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)

def average_slope_intercept(lines):
    left_lines, left_weights = [], []
    right_lines, right_weights = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))

def lane_area(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_points = pixel_points(y1, y2, left_lane)
    right_points = pixel_points(y1, y2, right_lane)

    if left_points and right_points:
        polygon_points = np.array([left_points[0], left_points[1], right_points[1], right_points[0]], np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        cv2.fillPoly(image, [polygon_points], (0, 255, 0))
    return image

def frame_processor(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    region = region_selection(edges)
    hough = hough_transform(region)
    return lane_area(image, hough)

# ========== Streamlit App ==========

st.title("Lane Detection WebApp")
st.write("Upload a road video and download the version with lane markings as polygons.")

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])



if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
        temp_input.write(uploaded_video.read())
        temp_input_path = temp_input.name

    st.video(temp_input_path)
    frame_display = st.empty()

    with st.spinner("Processing video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_output:
            output_path = temp_output.name

        cap = cv2.VideoCapture(temp_input_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_width, target_height = 960, 540
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use X264 codec for better compatibility
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        start_time = time.time()
        eta_placeholder = st.empty()
        progress_bar = st.progress(0, text="Initializing...")

        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (target_width, target_height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                processed_frame = frame_processor(frame_rgb)
                processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                out.write(processed_frame_bgr)
                processed_frames += 1

                # added by hardik
                frame_display.image(processed_frame, channels="RGB", caption=f"Frame {processed_frames + 1}")

                # ETA
                elapsed_time = time.time() - start_time
                remaining_frames = total_frames - processed_frames
                eta = (elapsed_time / processed_frames) * remaining_frames if processed_frames > 0 else 0
                eta_min, eta_sec = divmod(int(eta), 60)
                eta_placeholder.markdown(f"***Remaining Time: `{eta_min:02d}:{eta_sec:02d}` remaining***")

                # Progress bar
                progress_percent = processed_frames / total_frames
                progress_bar.progress(progress_percent, text=f"Processing {processed_frames}/{total_frames} frames...")

            except Exception as e:
                st.error(f"Error processing frame: {e}")
                break

        cap.release()
        out.release()

        total_time = time.time() - start_time
        total_min, total_sec = divmod(int(total_time), 60)

        st.success("Processing complete!!!!!!!!!")
        st.markdown(f"***Total Time Taken: `{total_min:02d}:{total_sec:02d}`***")
        st.video(output_path)  # Display the processed video

        with open(output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_output.mp4",  # File name for download
                mime="video/mp4"  # MIME type for mp4 video
            )

        os.remove(temp_input_path)
        os.remove(output_path)