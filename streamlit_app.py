import streamlit as st
import sys
import subprocess
import pkg_resources

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Verificar y instalar dependencias
required = {'ultralytics', 'Pillow'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    st.warning(f"Instalando dependencias faltantes: {', '.join(missing)}")
    for package in missing:
        install(package)
    st.rerun()

try:
    from ultralytics import YOLO
    from PIL import Image
except ImportError as e:
    st.error(f"Error al importar las bibliotecas necesarias: {e}")
    st.info("Por favor, asegúrate de que todas las dependencias estén instaladas correctamente.")
    st.stop()

# Cargar el modelo YOLOv8
@st.cache_resource
def load_model():
    try:
        return YOLO('yolov8n.pt')  # Usar 'yolov8n.pt' o el path a tu modelo personalizado
    except Exception as e:
        st.error(f"Error al cargar el modelo YOLOv8: {e}")
        return None

model = load_model()

if model is None:
    st.error("No se pudo cargar el modelo. Por favor, verifica la instalación y la ruta del modelo.")
    st.stop()

st.title('Detector de Objetos con YOLOv8')

# Usar st.camera_input en lugar de st.file_uploader
img_file_buffer = st.camera_input("Toma una foto")

if img_file_buffer is not None:
    try:
        image = Image.open(img_file_buffer)
        st.image(image, caption='Imagen capturada.', use_column_width=True)

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
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")

st.write("Nota: Si encuentras problemas, asegúrate de que todas las dependencias estén instaladas y que el modelo YOLOv8 esté en el directorio correcto.")
