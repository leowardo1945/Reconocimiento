import cv2
import mediapipe as mp
import os

nombre = "Tapaboca"
direccion = 'D:/LeoWin/Fotos'
carpeta = direccion +'/' + nombre

if not os.path.exists(carpeta):
    print('carpeta creada')
    os.makedirs(carpeta)
# contador inicializado
count = 0
detector = mp.solutions.face_detection # funcion de  detector
dibujo = mp.solutions.drawing_utils # funcion de dibujo

cap = cv2.VideoCapture('https://192.168.0.106:8080/video') # aqui se establece la camara

with detector.FaceDetection(min_detection_confidence=0.75) as rostros:
    while True:
        # lectura de video
        ret , frame = cap.read()
        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # DETECCION DE ROSTROS
        resultado = rostros.process(rgb)
        # filtro de seguridad
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro)
                al ,an, _ = frame.shape
                xi = rostro.location_data.relative_bounding_box.xmin
                yi = rostro.location_data.relative_bounding_box.ymin
                ancho = rostro.location_data.relative_bounding_box.width
                alto = rostro.location_data.relative_bounding_box.height
                xi = int(xi * an)
                yi = int(yi * al)
                ancho = int(ancho * an)
                alto = int(alto * al)
                xf = xi + ancho
                yf = yi + alto
                # EXTRAER PIXELES
                cara = frame[yi:yf, xi:xf]
                # REDIMENCION DE FOTOS
                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)
                # GUARDAR LAS IMAGENES CON CV2
                cv2.imwrite(carpeta + "/rostro_{}.jpg".format(count), cara)
                count = count + 1
        # mostrar fotogramas
        cv2.imshow("Reconocimiento", frame)
        # leer tecla de salida
        t = cv2.waitKey(1)
        if t == 27 or count >= 300:
            break
cap.release()
cv2.destroyAllWindows()
