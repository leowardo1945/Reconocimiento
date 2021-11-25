import cv2
import mediapipe as mp

detector = mp.solutions.face_detection
dibujo = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with detector.FaceDetection(min_detection_confidence=0.75) as rostros:

    while True:
        # lectura de la video captura
        ret, frame = cap.read()
        # eliminar error de movimiento
        frame = cv2.flip(frame, 1)
        # correccion de colores
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # deteccion de rostros
        resultado = rostros.process(rgb)
        if resultado.detections is not None:
            for rostro in resultado.detections:
                dibujo.draw_detection(frame, rostro, dibujo.DrawingSpec(color=(0,255,0)))
                for id, coordenadas in enumerate(resultado.detections):
                    al, an, c =frame.shape
                    x = coordenadas.location_data_relative_bounding_box.xmin
                    y = coordenadas.location_data_relative_bounding_box.ymin
                    ancho =  coordenadas.location_data_relative_bounding_width
                    alto = coordenadas.location_data_relative_bounding_height
                    xi, yi = int(x * an), int(y * al)
                    xf, yf = int(ancho * an), int(alto * al)


        cv2.imshow("camara",frame)
        t = cv2.waitKey(1)
        if t == 27:
            break
cap.release()
cv2.destroyAllWindows()