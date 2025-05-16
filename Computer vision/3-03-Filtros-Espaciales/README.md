[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)  [![OpenCV](https://img.shields.io/badge/opencv-4.x-blue)](https://opencv.org/)  [![NumPy](https://img.shields.io/badge/numpy-1.24%2B-blue)](https://numpy.org/)  [![Matplotlib](https://img.shields.io/badge/matplotlib-3.x-orange)](https://matplotlib.org/)

# 🧠 Tarea de Procesamiento de Video en Vivo con OpenCV

<!-- toc -->
## 📌 Índice
- [📘 Contenido de la Tarea](#contenido-de-la-tarea)  
- [1. 🎥 Captura de Video en Vivo](#1-captura-de-video-en-vivo)  
- [2. ⚙️ Aplicación de Filtros en Tiempo Real](#2-aplicación-de-filtros-en-tiempo-real)  
- [3. 🎮 Controles Interactivos](#3-controles-interactivos)  
- [4. ⌨️ Mapeo de Teclas y Flechas](#4-mapeo-de-teclas-y-flechas)  
- [📦 Dependencias](#dependencias)  
- [🗂️ Estructura del Proyecto](#estructura-del-proyecto)  
- [🔍 Referencias](#referencias)  
- [✍️ Autor y Fecha](#autor-y-fecha)  
<!-- tocstop -->

## 📘 Contenido de la Tarea
1. Captura de video en vivo  
2. Aplicación de filtros en tiempo real  
3. Controles interactivos (pause, salir)  
4. Mapeo de teclas y flechas para cambiar parámetros  

---

## 1. 🎥 Captura de Video en Vivo
```python
import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Live Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 2. ⚙️ Aplicación de Filtros en Tiempo Real 

```python
# Definición previa de funciones: filter_original, filter_average, filter_gaussian, …
filters = {
    '0': filter_original,
    '1': filter_average,
    '2': filter_gaussian,
    '3': filter_median,
    '4': filter_bilateral,
    '5': filter_laplacian,
    '6': filter_sobel,
    '7': filter_canny,
    '8': filter_hough_lines,
    '9': filter_hough_linesp,
    'a': filter_hough_circles,
}

# Trackbar para kernel
cv2.namedWindow("Control")
cv2.createTrackbar("Kernel","Control",5,31, lambda v: None)

active = '0'
t = 100  # umbral inicial
```

Dentro del bucle principal, se lee `k = cv2.getTrackbarPos("Kernel","Control")`, se ajusta a impar, y se aplica:

```python
out = filters[active](frame, k, t)
cv2.putText(out, f"Filtro {active} | k={k} | t={t}",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
cv2.imshow("Processed", out)
```

---

## 3. 🎮 Controles Interactivos

* **q**: salir del programa
* **Trackbar “Kernel”**: ajustar tamaño de ventana (de 1 a 31, solo impares)
* **Teclas 0–9, a**: seleccionar filtro activo

---

## 4. ⌨️ Mapeo de Teclas y Flechas

```python
key = cv2.waitKeyEx(1)

# Flechas ↑/↓ para umbral t
if key in (82, 2490368):      # ↑
    t = min(t + 5, 1000)
elif key in (84, 2621440):    # ↓
    t = max(t - 5, 0)
elif key == ord('q'):
    break
else:
    c = chr(key & 0xFF)
    if c in filters:
        active = c
```

---

## 📦 Dependencias

* Python 3.12
* OpenCV (`pip install opencv-contrib-python`)
* NumPy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)

---

## 🗂️ Estructura del Proyecto

```
video-filtros-opencv/
├── images/
├── screenshots/
├── 3-03-Filtros-Espaciales.ipynb
└── README.md
```

---

## 🔍 Referencias

* Ingeniería del Conocimiento UNCP 2025: [https://github.com/Jaime1406/Ingenieria\_del\_conocimiento\_UNCP\_2025/](https://github.com/Jaime1406/Ingenieria_del_conocimiento_UNCP_2025/)
* CS231n: Deep Learning for Computer Vision: [https://cs231n.stanford.edu/](https://cs231n.stanford.edu/)

---

## ✍️ Autor y Fecha

* **Autor:** Jhulio Alessandro Morán de La Cruz
* **Github:** [@jhulio435m](https://github.com/jhulio435m)
* **Fecha**: 16 de mayo de 2025
