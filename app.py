import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Deteccion de Objetos en Tiempo Real", page_icon="\U0001F4F7", layout="wide")

st.title("Deteccion de Objetos con IA (YOLO)")
st.write("Detecta objetos en imagenes o video usando un modelo YOLO preentrenado.")


@st.cache_resource
def cargar_modelo():
    return YOLO("yolov8n.pt")


modelo = cargar_modelo()

with st.sidebar:
    st.header("Configuracion")
    confianza = st.slider("Umbral de confianza", 0.1, 1.0, 0.4, 0.05)
    modo = st.radio("Fuente de entrada", ["Imagen", "Camara (foto)"])

imagen_entrada = None

if modo == "Imagen":
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if archivo:
        imagen_entrada = Image.open(archivo).convert("RGB")
else:
    foto = st.camera_input("Toma una foto")
    if foto:
        imagen_entrada = Image.open(foto).convert("RGB")

if imagen_entrada is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(imagen_entrada, use_container_width=True)

    with st.spinner("Detectando objetos..."):
        resultados = modelo.predict(np.array(imagen_entrada), conf=confianza, verbose=False)
        imagen_anotada = resultados[0].plot()
        imagen_anotada = cv2.cvtColor(imagen_anotada, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("Objetos detectados")
        st.image(imagen_anotada, use_container_width=True)

    detecciones = resultados[0].boxes
    if detecciones is not None and len(detecciones) > 0:
        st.subheader("Resumen de detecciones")
        nombres = modelo.names
        conteo = {}
        for clase_id in detecciones.cls.tolist():
            nombre = nombres[int(clase_id)]
            conteo[nombre] = conteo.get(nombre, 0) + 1
        for nombre, cantidad in conteo.items():
            st.write(f"- {nombre}: {cantidad}")
    else:
        st.info("No se detectaron objetos con el umbral de confianza actual.")
else:
    st.info("Sube una imagen o toma una foto para comenzar la deteccion.")
