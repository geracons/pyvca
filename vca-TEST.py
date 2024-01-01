import numpy as np
import cv2
import time
import logging
from tkinter import filedialog
import tkinter as tk
import os

# Función para obtener el archivo de video mediante una interfaz gráfica
def obtener_archivo_video():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal

    archivo_video = filedialog.askopenfilename(title="Seleccionar archivo de video", filetypes=[("Archivos de video", "*.mp4;*.avi;*.mkv")])

    return archivo_video

# Función para obtener la velocidad del video
def obtener_velocidad():
    velocidad = float(input("Ingrese la velocidad del video (1.0 para reproducción normal): "))
    return velocidad

# Configuración de logging
logging.basicConfig(filename='detecciones.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Obtener el archivo de video
archivo_video = obtener_archivo_video()

# Obtener la velocidad del video
velocidad_video = obtener_velocidad()

# Cargamos el vídeo
camara = cv2.VideoCapture(archivo_video)

# Establecer la velocidad del video
camara.set(cv2.CAP_PROP_FPS, camara.get(cv2.CAP_PROP_FPS) * velocidad_video)

# Inicializamos el primer frame a vacío.
# Nos servirá para obtener el fondo
fondo = None

# Umbral para el área del contorno
umbral_area_contorno = 1000

# Intervalo de tiempo entre detecciones en segundos
intervalo_entre_detecciones = 4

# Último tiempo de detección registrado
ultimo_tiempo_deteccion = 0

# Directorio para guardar las capturas
directorio_capturas = 'capturas'
os.makedirs(directorio_capturas, exist_ok=True)

# Obtener el nombre del archivo sin la ruta
nombre_archivo = os.path.basename(archivo_video)

# Agregar el nombre del archivo al registro de log
logging.info(f'Archivo de video seleccionado: {nombre_archivo}, Velocidad del video: {velocidad_video}')

# Recorremos todos los frames
while True:
    # Obtenemos el frame
    (grabbed, frame) = camara.read()

    # Si hemos llegado al final del vídeo salimos
    if not grabbed:
        break

    # Convertimos a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicamos suavizado para eliminar ruido
    gris = cv2.GaussianBlur(gris, (21, 21), 0)

    # Si todavía no hemos obtenido el fondo, lo obtenemos
    # Será el primer frame que obtengamos
    if fondo is None:
        # Permitir al usuario seleccionar una región de interés (ROI) en el primer frame
        rect_roi = cv2.selectROI(frame, False)
        cv2.destroyAllWindows()  # Cerrar la ventana después de la selección
        (x, y, w, h) = rect_roi
        fondo = gris[y:y + h, x:x + w].copy()
        continue

    # Ajustamos las dimensiones de la región de interés
    fondo_roi = cv2.resize(fondo, (w, h))

    # Calculo de la diferencia entre el fondo y el frame actual
    resta = cv2.absdiff(fondo_roi, gris[y:y + h, x:x + w])

    # Aplicamos un umbral
    umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilatamos el umbral para tapar agujeros
    umbral = cv2.dilate(umbral, None, iterations=2)

    # Copiamos el umbral para detectar los contornos
    contornosimg = umbral.copy()

    # Find contours
    if cv2.__version__.startswith('3.'):
        _, contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif cv2.__version__.startswith('4.'):
        contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Lógica de manejo para otras versiones de OpenCV
        pass

    # Resto del código usando 'contours' y 'hierarchy' según sea necesario

    # Recorremos todos los contornos encontrados
    for c in contornos:
        # Eliminamos los contornos más pequeños
        if cv2.contourArea(c) < umbral_area_contorno:
            continue

        # Obtenemos el bounds del contorno, el rectángulo mayor que engloba al contorno
        (x_contorno, y_contorno, w_contorno, h_contorno) = cv2.boundingRect(c)

        # Ajustamos las coordenadas con respecto al ROI seleccionado
        x_contorno += x
        y_contorno += y

        # Dibujamos el rectángulo del bounds
        cv2.rectangle(frame, (x_contorno, y_contorno), (x_contorno + w_contorno, y_contorno + h_contorno), (0, 255, 0), 2)

        # Obtenemos el tiempo exacto del video en milisegundos
        tiempo_actual_milisegundos = camara.get(cv2.CAP_PROP_POS_MSEC)

        # Convertimos el tiempo a segundos
        tiempo_actual_segundos = tiempo_actual_milisegundos / 1000.0

        # Verificamos si ha pasado el intervalo de tiempo desde la última detección
        if tiempo_actual_segundos - ultimo_tiempo_deteccion >= intervalo_entre_detecciones:
            # Generamos un registro de log con el tiempo del video, área del contorno y otros datos
            logging.info(f'Tiempo del video: {tiempo_actual_segundos:.2f} segundos, Detección en el frame: {camara.get(cv2.CAP_PROP_POS_FRAMES)}, '
                         f'Área del contorno: {cv2.contourArea(c)}')

            # Actualizamos el último tiempo de detección registrado
            ultimo_tiempo_deteccion = tiempo_actual_segundos

            # Guardamos la captura como imagen PNG
            nombre_captura = f'{directorio_capturas}/captura_{camara.get(cv2.CAP_PROP_POS_FRAMES)}_{tiempo_actual_segundos:.2f}.png'
            cv2.imwrite(nombre_captura, frame)

    # Mostramos las imágenes de la cámara, el umbral y la resta
    cv2.imshow("Camara", frame)
    cv2.imshow("Umbral", umbral)
    cv2.imshow("Resta", resta)
    cv2.imshow("Contorno", contornosimg)

    # Capturamos una tecla para salir
    key = cv2.waitKey(1) & 0xFF

    # Si ha pulsado la letra s, salimos
    if key == ord("s"):
        break

# Liberamos la cámara y cerramos todas las ventanas
camara.release()
cv2.destroyAllWindows()
