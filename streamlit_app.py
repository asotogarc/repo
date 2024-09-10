import streamlit as st
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
res = model.predict(file,conf=0.5,save=True)
box = res[0].boxes.xyxy.tolist()
res_plotted = res[0].plot()[:, :, ::-1]
st.image(res_plotted, caption='Text Detections',use_column_width=True)
st.write("Number of the Detections : "+str(len(box)))
