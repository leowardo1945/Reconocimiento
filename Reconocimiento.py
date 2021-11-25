import cv2
import os
import mediapipe as mp

direccion = "D:/LeoWin/Fotos"
etiquetas = os.listdir(direccion)
print("Nombres",etiquetas)

# llamar modelo entrenado
modelo = cv2.face.LBPHFaceRecognizer_create()
# leer modelo
modelo.read('ModeloEntrenado.xml')
detector = mp.solutions.face_detection # funcion de  detector
dibujo = mp.solutions.drawing_utils # funcion de dibujo

cap = cv2.VideoCapture('https://192.168.0.106:8080/video') # aqui se establece la camara

with detector.FaceDetection(min_detection_confidence=0.75) as rostros:
    while True:
        # lectura de video
        ret , frame = cap.read()
        copia = frame.copy()
        frame = cv2.flip(copia,1)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        copia2 = rgb.copy()
        # DETECCION DE ROSTROS
        resultado = rostros.process(copia2)
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
                cara = copia2[yi:yf, xi:xf]
                # REDIMENCION DE FOTOS
                cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)
                cara = cv2.cvtColor(cara, cv2.COLOR_BGR2GRAY)
                #REALIZAR PREDICCION
                prediccion = modelo.predict(cara)
                # mostrar resultados en pantalla
                if prediccion[0] == 0:
                    cv2.putText(frame,'{}'.format(etiquetas[0]),(xi,yi -5), 1, 1.3, (0,0,255), 1 , cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf,yf), (0,0,255), 2)
                if prediccion[0] == 1:
                    cv2.putText(frame,'{}'.format(etiquetas[1]),(xi,yi -5), 1, 1.3, (255,0,0), 1 , cv2.LINE_AA)
                    cv2.rectangle(frame, (xi,yi), (xf,yf), (255,0,0), 2)

        # mostrar fotogramas
        cv2.imshow("Reconocimiento", frame)
        # leer tecla de salida
        t = cv2.waitKey(1)
        if t == 27:
            break
cap.release()
cv2.destroyAllWindows()