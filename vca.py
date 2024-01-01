import numpy as np
import cv2
import time
import logging
from tkinter import filedialog
import tkinter as tk
import os

# SELECTOR ARCHIVO
def obtener_archivo_video():
    root = tk.Tk()
    root.withdraw()  

    archivo_video = filedialog.askopenfilename(title="Seleccionar archivo de video", filetypes=[("Archivos de video", "*.mp4;*.avi;*.mkv")])

    return archivo_video

# VELOCIDAD VIDEO
def obtener_velocidad():
    velocidad = float(input("Ingrese la velocidad del video (1.0 normal): "))
    return velocidad

# LOGGING
logging.basicConfig(filename='detecciones.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# OBT VIDEO FILE
archivo_video = obtener_archivo_video()

# OBT VIDEO VEL
velocidad_video = obtener_velocidad()

# VIDEO CHARGE
camara = cv2.VideoCapture(archivo_video)

# SET VIDEO VEL
camara.set(cv2.CAP_PROP_FPS, camara.get(cv2.CAP_PROP_FPS) * velocidad_video)


fondo = None

# UMBRAL ENTORNO - SE PUEDE MODIFICAR EN BASE A LA NECESIDAD DE DETECCION
umbral_area_contorno = 1000

# INTERVALO DE TIEMPO ENTRE DETECCIONES
intervalo_entre_detecciones = 4

# ULTIMO TIEMPO DE DETECCION REGISTRADO
ultimo_tiempo_deteccion = 0

# PATH CAPTURAS
directorio_capturas = 'capturas'
os.makedirs(directorio_capturas, exist_ok=True)

# GET FILENAME WO PATH
nombre_archivo = os.path.basename(archivo_video)

# ADD FILENAME TO LOG
logging.info(f'Archivo de video seleccionado: {nombre_archivo}, Velocidad del video: {velocidad_video}')


while True:
    (grabbed, frame) = camara.read()
    if not grabbed:
        break

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gris = cv2.GaussianBlur(gris, (21, 21), 0)

    if fondo is None:
        # ROI
        rect_roi = cv2.selectROI(frame, False)
        cv2.destroyAllWindows() 
        (x, y, w, h) = rect_roi
        fondo = gris[y:y + h, x:x + w].copy()
        continue

    # ROI ADJUST
    fondo_roi = cv2.resize(fondo, (w, h))

    resta = cv2.absdiff(fondo_roi, gris[y:y + h, x:x + w])

    umbral = cv2.threshold(resta, 25, 255, cv2.THRESH_BINARY)[1]

 
    umbral = cv2.dilate(umbral, None, iterations=2)


    contornosimg = umbral.copy()

    if cv2.__version__.startswith('3.'):
        _, contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif cv2.__version__.startswith('4.'):
        contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
     
        pass

 
    for c in contornos:

        if cv2.contourArea(c) < umbral_area_contorno:
            continue

      
        (x_contorno, y_contorno, w_contorno, h_contorno) = cv2.boundingRect(c)

    
        x_contorno += x
        y_contorno += y

      
        cv2.rectangle(frame, (x_contorno, y_contorno), (x_contorno + w_contorno, y_contorno + h_contorno), (0, 255, 0), 2)


        tiempo_actual_milisegundos = camara.get(cv2.CAP_PROP_POS_MSEC)

        tiempo_actual_segundos = tiempo_actual_milisegundos / 1000.0

        
        if tiempo_actual_segundos - ultimo_tiempo_deteccion >= intervalo_entre_detecciones:
          
            logging.info(f'Tiempo del video: {tiempo_actual_segundos:.2f} segundos, Detección en el frame: {camara.get(cv2.CAP_PROP_POS_FRAMES)}, '
                         f'Área del contorno: {cv2.contourArea(c)}')

          
            ultimo_tiempo_deteccion = tiempo_actual_segundos

           
            nombre_captura = f'{directorio_capturas}/captura_{camara.get(cv2.CAP_PROP_POS_FRAMES)}_{tiempo_actual_segundos:.2f}.png'
            cv2.imwrite(nombre_captura, frame)

   
    cv2.imshow("Camara", frame)
    

   
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        break


camara.release()
cv2.destroyAllWindows()
