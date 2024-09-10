import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Streamlit app
st.title("Object Detection with YOLO")

# Camera input
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Read image file buffer with OpenCV
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Perform YOLO prediction
    results = model.predict(cv2_img, conf=0.5)

    # Get bounding boxes
    boxes = results[0].boxes.xyxy.tolist()

    # Plot results
    res_plotted = results[0].plot()[:, :, ::-1]

    # Display the image with detections
    st.image(res_plotted, caption='Object Detections', use_column_width=True)

    # Display number of detections
    st.write(f"Number of Detections: {len(boxes)}")
