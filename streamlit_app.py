import streamlit as st
from ultralytics.solutions import streamlit_inference as stui

def main():
    st.title("Detección de Objetos en Tiempo Real con YOLOv8")
    st.write("Esta aplicación utiliza YOLOv8 para detectar objetos en tiempo real desde tu cámara.")

    # Configuración del modelo
    model_path = "yolov8n.pt"  # Modelo YOLOv8 nano por defecto

    # Llamada a la función de inferencia de Ultralytics
    stui.inference(model=model_path)

    st.write("Instrucciones:")
    st.write("1. Haz clic en 'Start' para iniciar la detección en tiempo real.")
    st.write("2. Permite el acceso a tu cámara cuando se te solicite.")
    st.write("3. Ajusta los parámetros en el panel lateral si lo deseas.")
    st.write("4. Haz clic en 'Stop' para detener la detección.")

if __name__ == "__main__":
    main()
