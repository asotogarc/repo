import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Cargar el modelo YOLOv8
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Usar 'yolov8n.pt' o el path a tu modelo personalizado

model = load_model()

st.title('Detector de Objetos con YOLOv8')

uploaded_image = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Imagen subida.', use_column_width=True)

    if st.button('Detectar Objetos'):
        results = model(image)
        
        # Visualizar resultados
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            st.image(im, caption='Resultado de la detección.', use_column_width=True)

        # Mostrar resultados en formato de texto
        for result in results:
            boxes = result.boxes.data.tolist()
            st.write("Objetos detectados:")
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box
                st.write(f"Clase: {result.names[int(class_id)]}, Confianza: {score:.2f}")

st.write("Nota: Asegúrate de tener instaladas las bibliotecas necesarias (streamlit, ultralytics, pillow) y un modelo YOLOv8 en el mismo directorio que este script.")
