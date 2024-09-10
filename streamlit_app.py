import streamlit as st
from PIL import Image
import numpy as np
import torch

# Importing YOLO directly from torch hub to avoid OpenCV dependency
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def main():
    st.title("Detección de Objetos con YOLOv5")
    
    # Usar st.camera_input para capturar una imagen de la webcam
    img_file_buffer = st.camera_input("Toma una foto")
    
    if img_file_buffer is not None:
        # Convertir la imagen a un objeto PIL Image
        image = Image.open(img_file_buffer)
        
        # Realizar la detección de objetos
        results = model(image)
        
        # Obtener la imagen con las detecciones
        img_with_boxes = Image.fromarray(results.render()[0])
        
        # Mostrar la imagen con las detecciones
        st.image(img_with_boxes, caption="Imagen con Detecciones", use_column_width=True)
        
        # Mostrar las clases detectadas
        detected_classes = results.pandas().xyxy[0]['name'].unique()
        st.write("Objetos detectados:", ', '.join(detected_classes))
    
    else:
        st.info("Esperando a que tomes una foto...")

if __name__ == "__main__":
    main()
