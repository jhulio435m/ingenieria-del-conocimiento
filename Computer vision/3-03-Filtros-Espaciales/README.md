# Tarea de Procesamiento de Video en Vivo con OpenCV

Esta tarea utiliza el notebook **`3-03-Filtros-Espaciales.ipynb`**, en el cual se captura y procesa video en tiempo real desde una cámara web usando OpenCV y NumPy.

---

## 📋 Contenido

1. **Captura de video en vivo**
2. **Aplicación de filtros en tiempo real**
3. **Controles interactivos**
4. **Mapeo de teclas y flechas**

---

## ⚙️ Requisitos

* Python 3.7+
* OpenCV (`opencv--contrib-python`)
* Matplotlib
* NumPy

Instalación:

```bash
pip install opencv-contrib-python numpy matplotlib
```

---

## 🚀 Uso

1. Abre **Jupyter Notebook** y carga `3-03-Filtros-Espaciales.ipynb`.
2. Ejecuta la celda de **Captura en vivo**.
3. Ajusta el tamaño de kernel con el **trackbar**.
4. Cambia de filtro con las teclas indicadas.
5. Modifica el umbral `t` con las flechas ↑/↓.

---

## 🎥 Código de Captura en Vivo

```python
import cv2
import numpy as np

# Definición de filtros
# Cada función recibe: frame, kernel k, umbral t
# ... (implementaciones idénticas a las del notebook) ...

# Mapeo de filtros a teclas
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

# Inicializar cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

# Crear ventana de control
cv2.namedWindow("Control")
cv2.createTrackbar("Kernel","Control",5,31, lambda v: None)

active = '0'
t = 100  # umbral inicial

while True:
    ret, frame = cap.read()
    if not ret: break

    # Leer kernel y asegurar impar
    k = cv2.getTrackbarPos("Kernel","Control")
    k = k if k%2==1 else k+1

    # Aplicar filtro activo
    out = filters[active](frame, k, t)
    cv2.putText(out, f"Filtro {active} | k={k} | t={t}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Processed", out)

    key = cv2.waitKeyEx(1)
    if key in (82, 2490368):  # flecha arriba
        t = min(t+5, 1000)
    elif key in (84, 2621440):  # flecha abajo
        t = max(t-5, 0)
    elif key == ord('q'):
        break
    else:
        c = chr(key & 0xFF)
        if c in filters:
            active = c

cap.release()
cv2.destroyAllWindows()
```

---

## 🔍 Filtros Disponibles

| Tecla | Filtro           |
| ----- | ---------------- |
| 0     | Original         |
| 1     | Promedio         |
| 2     | Gaussiano        |
| 3     | Mediana          |
| 4     | Bilateral        |
| 5     | Laplaciano       |
| 6     | Sobel (magnitud) |
| 7     | Canny            |
| 8     | HoughLines       |
| 9     | HoughLinesP      |
| a     | HoughCircles     |

---

## Referencias

* Ingenieria_del_conocimiento_UNCP_2025: [https://github.com/Jaime1406/Ingenieria_del_conocimiento_UNCP_2025/](https://github.com/Jaime1406/Ingenieria_del_conocimiento_UNCP_2025/)
* CS231n: Deep Learning for Computer Vision: [https://cs231n.stanford.edu//](https://cs231n.stanford.edu//)

---

## 🎮 Controles

* **Flechas ←/→**: Ajustan el tamaño de ventana `k` (kernel) usado por los filtros de suavizado y Sobel; ← para disminuir y → para aumentar.
* **Flechas ↑/↓**: Incrementan o decrementan el umbral `t`.
* **Teclas 0–9, a**: Seleccionan el filtro activo.
* **q**: Sale del programa.

---

© 2025 Jhulio Alessandro Morán de la Cruz

### Fecha: 16 de mayo de 2025


