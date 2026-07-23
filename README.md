# Deteccion de Objetos en Tiempo Real con IA

Aplicacion de deteccion de objetos usando el modelo YOLOv8, capaz de analizar imagenes o fotos capturadas por camara.

## Descripcion

Esta aplicacion permite subir una imagen o tomar una foto con la camara para detectar y clasificar objetos presentes en ella usando un modelo YOLO preentrenado. Muestra la imagen original junto con la imagen anotada, ademas de un resumen con la cantidad de objetos detectados por clase.

## Caracteristicas

- Deteccion de objetos mediante YOLOv8 (Ultralytics).
- Soporte para carga de imagenes o captura por camara.
- Umbral de confianza ajustable.
- Visualizacion comparativa: imagen original vs. imagen anotada.
- Resumen de objetos detectados por categoria.
- Interfaz simple e intuitiva construida con Streamlit.

## Tecnologias utilizadas

- Python
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pillow

## Instalacion

1. Clonar el repositorio:
```
git clone https://github.com/Ja1ros/deteccion-objetos-tiempo-real.git
cd deteccion-objetos-tiempo-real
```

2. Instalar dependencias:
```
pip install -r requirements.txt
```

El modelo YOLOv8 (yolov8n.pt) se descargara automaticamente en la primera ejecucion.

## Uso

Ejecuta la aplicacion con:
```
streamlit run app.py
```

Luego abre tu navegador en `http://localhost:8501`, sube una imagen o toma una foto y observa los objetos detectados en tiempo real.

## Licencia

Este proyecto es de uso libre para fines educativos y de portafolio.
