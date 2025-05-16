[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  [![OpenCV](https://img.shields.io/badge/opencv-4.7.0-blue)](https://opencv.org/)  [![NumPy](https://img.shields.io/badge/numpy-1.24%2B-blue)](https://numpy.org/)  [![Matplotlib](https://img.shields.io/badge/matplotlib-3.x-orange)](https://matplotlib.org/)

# 🧠 Tarea Práctica: Procesamiento Básico de Imágenes con OpenCV

<!-- toc -->
## 📌 Índice
- [📘 Contenido de la Tarea](#📘-contenido-de-la-tarea)  
- [1. 📷 Lectura y Visualización de Imágenes](#1-📷-lectura-y-visualización-de-imágenes)  
- [2. 📊 Cálculo y Visualización de Histogramas](#2-📊-cálculo-y-visualización-de-histogramas)  
- [3. 🔄 Comparación de Histogramas Antes/Después](#3-🔄-comparación-de-histogramas-antesdespués)  
- [4. ⚪ Umbralización](#4-⚪-umbralización)  
- [5. 🧮 Morfología Matemática](#5-🧮-morfología-matemática)  
- [6. 🔀 Transformaciones Espaciales](#6-🔀-transformaciones-espaciales)  
- [7. 🌐 Transformada de Fourier](#7-🌐-transformada-de-fourier)  
- [8. 🎨 Procesamiento de Color](#8-🎨-procesamiento-de-color)  
- [9. 📏 Métricas de Calidad (MSE & PSNR)](#9-📏-métricas-de-calidad-mse--psnr)  
- [10.🔪 Filtro Sharpen](#10-🔪-filtro-sharpen)
- [📦 Dependencias](#📦-dependencias)  
- [🗂️ Estructura del Proyecto](#🗂️-estructura-del-proyecto)  
- [🔍 Referencias](#🔍-referencias)  
- [✍️ Autor y Fecha](#✍️-autor-y-fecha)  
<!-- tocstop -->

## 📘 Contenido de la Tarea
1. Lectura y visualización de imágenes  
2. Cálculo y visualización de histogramas  
3. Comparación de histogramas antes/después de aplicación de filtros  
4. Umbralización (fijo y Otsu)  
5. Operaciones de morfología matemática  
6. Transformaciones espaciales (traslación, rotación, escala, perspectiva)  
7. Transformada de Fourier y filtrado en frecuencia  
8. Procesamiento de color (HSV, segmentación, máscaras)  
9. Cálculo de métricas de calidad: MSE y PSNR  

---

## 1. 📷 Lectura y Visualización de Imágenes
```python
def cargar_imagen(ruta: str) -> np.ndarray:
    img_bgr = cv2.imread(ruta)
    if img_bgr is None:
        raise FileNotFoundError(f"No se encontró la imagen: {ruta}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def mostrar_imagen(img: np.ndarray, título: str = "Imagen", tamaño=(8,6)):
    plt.figure(figsize=tamaño)
    plt.imshow(img)
    plt.title(título)
    plt.axis('off')
    plt.show()

# — Ejemplo —
img = cargar_imagen('images/sideral.png')
mostrar_imagen(img, 'Original')
print(f"Dimensiones: {img.shape}")
```
### Resultado

```bash
Dimensiones de la imagen: (225, 225, 3)
```

**Captura de Pantalla**

![Lectura y visualización](screenshots/captura_1.png)

---

## 2. 📊 Cálculo y Visualización de Histogramas

```python
def calcular_histograma(img: np.ndarray, canal:int, bins:int=256):
    data = img[:,:,canal].ravel()
    hist, edges = np.histogram(data, bins=bins, range=(0,256))
    return hist, edges

def mostrar_histograma(img: np.ndarray, canales=[0,1,2],
                       labels=['Rojo','Verde','Azul'],
                       título='Histograma RGB', tamaño=(8,5)):
    plt.figure(figsize=tamaño)
    for c,label in zip(canales, labels):
        h,e = calcular_histograma(img, c)
        plt.plot(e[:-1], h, label=label)
    plt.title(título); plt.xlabel('Intensidad'); plt.ylabel('Frecuencia')
    plt.legend(); plt.grid(True); plt.show()

# — Ejemplo —
mostrar_histograma(img)
```
### Resultado

**Captura de Pantalla**

![Histograma](screenshots/captura_2.png)

---

## 3. 🔄 Comparación de Histogramas Antes/Después

```python
def suavizar_gauss(img: np.ndarray, k=(5,5), sigma=0) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    suav = cv2.GaussianBlur(bgr, k, sigmaX=sigma)
    return cv2.cvtColor(suav, cv2.COLOR_BGR2RGB)

# Mostrar original vs suavizada y sus histogramas
img_suav = suavizar_gauss(img, (7,7), 1.5)
```
### Resultado

**Captura de Pantalla**

![Comparación de histogramas](screenshots/captura_3.png)

---

## 4. ⚪ Umbralización

```python
def umbral_fijo(img_gray: np.ndarray, thresh=127, max_v=255):
    _, binar = cv2.threshold(img_gray, thresh, max_v, cv2.THRESH_BINARY)
    return binar

def umbral_otsu(img_gray: np.ndarray):
    t, binar = cv2.threshold(img_gray, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binar, int(t)

# — Ejemplo —
gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```
### Resultado

**Captura de Pantalla**

![Umbralización](screenshots/captura_4.png)

---

## 5. 🧮 Morfología Matemática

```python
def crear_kernel(shape='rect', ksize=(5,5)):
    maps = {'rect':cv2.MORPH_RECT, 'ellipse':cv2.MORPH_ELLIPSE,
            'cross':cv2.MORPH_CROSS}
    return cv2.getStructuringElement(maps[shape], ksize)

def oper_morf(img_bin, kernel, op):
    return cv2.morphologyEx(img_bin, op, kernel)

# — Ejecución de ejemplo —
_, img_bin = cv2.threshold(gris,0,255,
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kern = crear_kernel('ellipse',(7,7))
```
### Resultado

**Captura de Pantalla**

![Morfología matemática](screenshots/captura_5.png)

---

## 6. 🔀 Transformaciones Espaciales

```python
def trasladar(img, tx, ty):
    h,w = img.shape[:2]
    M = np.array([[1,0,tx],[0,1,ty]],dtype=np.float32)
    bgr = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    res = cv2.warpAffine(bgr, M, (w,h))
    return cv2.cvtColor(res,cv2.COLOR_BGR2RGB)

def rotar(img, angle, scale=1.0, center=None):
    h,w = img.shape[:2]
    if center is None: center=(w//2,h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    bgr = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    res = cv2.warpAffine(bgr, M, (w,h))
    return cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
```
### Resultado

**Captura de Pantalla**

![Transformaciones espaciales](screenshots/captura_6.png)

---

## 7. 🌐 Transformada de Fourier

```python
def calcular_fft(img_gray):
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    mag = 20*np.log(np.abs(fshift)+1)
    return fshift, mag
```
### Resultado

**Captura de Pantalla**

![Transformada de Fourier](screenshots/captura_7.png)

---

## 8. 🎨 Procesamiento de Color

```python
def hsv_segmentación(img_bgr, lower, upper):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
```
### Resultado

**Captura de Pantalla**

![Procesamiento de color](screenshots/captura_8.png)

---

## 9. 📏 Métricas de Calidad (MSE & PSNR)

```python
def mse(img1, img2):
    return np.mean((img1.astype(float)-img2.astype(float))**2)

def psnr(img1, img2, max_v=255.0):
    e = mse(img1,img2)
    return float('inf') if e==0 else 20*np.log10(max_v/np.sqrt(e))
```
### Resultado
```bash
MSE Original vs Pasa Bajos: 222.72
PSNR Original vs Pasa Bajos: 24.65 dB

```

## 10. 🔪 Filtro Sharpen

```python
def filtro_sharpen(img_rgb: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]], dtype=np.float32)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    sharpen_bgr = cv2.filter2D(img_bgr, ddepth=-1, kernel=kernel)
    return cv2.cvtColor(sharpen_bgr, cv2.COLOR_BGR2RGB)

# — Ejecución de ejemplo —
img = cargar_imagen('images/sideral.png')
img_sharp = filtro_sharpen(img)
```

### Resultado
```bash
PSNR Original vs Sharpen: 20.63 dB

```
**Captura de Pantalla**
![MSE y PSNR](screenshots/captura_9.png)

---

## 📦 Dependencias

* Python 3.x
* OpenCV (`pip install opencv-contrib-python`)
* NumPy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)

---

## 🗂️ Estructura del Proyecto

```
procesamiento-imagenes-opencv/
├── images/               
│   ├── sideral.png
│   ├── animales.png
│   └── ...
├── screenshots/          
│   ├── captura_1.png
│   ├── ...
│   └── captura_10.png
├── 3-02-Procesamiento-Basico.ipynb
└── README.md
```

---

## 🔍 Referencias

* Ingeniería del Conocimiento UNCP 2025: [https://github.com/Jaime1406/Ingenieria\_del\_conocimiento\_UNCP\_2025/](https://github.com/Jaime1406/Ingenieria_del_conocimiento_UNCP_2025/)
* CS231n: Deep Learning for Computer Vision: [https://cs231n.stanford.edu/](https://cs231n.stanford.edu/)

---

## ✍️ Autor y Fecha

* 👤 **Jhulio Alessandro Morán de la Cruz**
* 🧑‍💻 [@jhulio435m](https://github.com/jhulio435m)
* 📅 **Fecha**: 16 de mayo de 2025