import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Carga el modelo YOLOv8 nano

def main():
    st.title("Detección de Objetos con YOLOv8")
    
    model = load_model()
    
    # Usar st.camera_input para capturar una imagen de la webcam
    img_file_buffer = st.camera_input("Toma una foto")
    
    if img_file_buffer is not None:
        # Convertir la imagen a un objeto PIL Image
        image = Image.open(img_file_buffer)
        
        # Convertir la imagen PIL a un array de numpy
        img_array = np.array(image)
        
        # Realizar la detección de objetos
        results = model(img_array)
        
        # Visualizar los resultados
        for r in results:
            im_array = r.plot()  # Plot los resultados con las cajas delimitadoras
            im = Image.fromarray(im_array[..., ::-1])  # Convertir de BGR a RGB
        
        # Mostrar la imagen con las detecciones
        st.image(im, caption="Imagen con Detecciones", use_column_width=True)
        
        # Mostrar las clases detectadas
        detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
        st.write("Objetos detectados:", ', '.join(set(detected_classes)))
    
    else:
        st.info("Esperando a que tomes una foto...")

if __name__ == "__main__":
    main()
