
# Tarea Práctica: Lectura, Escritura y Captura de Video con OpenCV
---

## Contenido de la tarea

1. Lectura y visualización de imágenes.
2. Mostrar imagen usando `cv2.imshow`.
3. Grabación de video capturado en archivo AVI.
4. Mejora interactiva: alternar color/escala de grises, pausar, salir, y selección de canales (rojo, verde, azul).
5. Espacio para capturas de pantalla de cada ejercicio.

---

## 1. Lectura y visualización de imagen

Se lee una imagen desde disco y se muestra su tamaño (dimensiones y canales).

```python
import cv2

I = cv2.imread('osin.jpeg')
print("Tamaño de la imagen BGR: ", I.shape)
cv2.imshow('Imagen Original', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 2. Mostrar imagen con `cv2.imshow`

Mostrar la imagen cargada hasta que se presiona cualquier tecla.

```python
cv2.imshow('Mostrar Imagen', I)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 3. Captura de video desde cámara

Capturar frames desde la cámara y mostrar video en color hasta que se presiona `q` para salir.

```python
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Video en Color', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## 4. Grabación de video desde cámara

Guardar el video capturado en archivo `output.avi` usando codec XVID y 20 FPS.

```python
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('Grabando Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

## 5. Mejora interactiva: modo color/escala de grises/canales RGB, pausa y selección de canal

Funciones para:

* `c`: alternar entre video en color y escala de grises.
* `p`: pausar y reanudar la captura.
* `q`: salir de la aplicación.
* `r`, `g`, `b`: mostrar sólo el canal rojo, verde o azul respectivamente, o volver a color completo si se presiona dos veces la misma tecla.

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
modo_gris = False
pausa = False
canal_color = None  # 'r', 'g', 'b' o None para color completo

while True:
    if not pausa:
        ret, frame = cap.read()
        if not ret:
            break

        if modo_gris:
            frame_mostrar = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif canal_color is not None:
            canales = {'b': 0, 'g': 1, 'r': 2}
            canal_idx = canales[canal_color]
            frame_mostrar = np.zeros_like(frame)
            frame_mostrar[:, :, canal_idx] = frame[:, :, canal_idx]
        else:
            frame_mostrar = frame

        cv2.imshow('Captura Cámara - c: Color/Gris, p: Pausa, q: Salir, r/g/b: Canales', frame_mostrar)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('c'):
        modo_gris = not modo_gris
        if modo_gris:
            canal_color = None
    elif key == ord('p'):
        pausa = not pausa
    elif key == ord('q'):
        break
    elif key in [ord('r'), ord('g'), ord('b')]:
        tecla = chr(key)
        if canal_color == tecla:
            canal_color = None
        else:
            canal_color = tecla
            modo_gris = False

cap.release()
cv2.destroyAllWindows()
```

---

## Requisitos

* Python 3.x instalado.
* Biblioteca OpenCV instalada (`pip install opencv-python`).

---
