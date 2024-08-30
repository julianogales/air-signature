import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inicializa la cámara
cap = cv2.VideoCapture(0)

# Inicializa el módulo de manos
with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    
    # Lista para almacenar los puntos de escritura
    points = []
    drawing_color = (0, 255, 0)  # Color del trazo (verde)
    drawing_thickness = 3  # Grosor del trazo

    def is_pinch(hand_landmarks, image_shape):
        # Obtén las coordenadas de los puntos clave del pulgar y el índice
        thumb_tip = hand_landmarks.landmark[4]
        index_finger_tip = hand_landmarks.landmark[8]
        h, w, _ = image_shape
        thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_finger_tip_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
        
        # Calcula la distancia entre la punta del pulgar y la punta del índice
        distance = math.hypot(index_finger_tip_coords[0] - thumb_tip_coords[0],
                              index_finger_tip_coords[1] - thumb_tip_coords[1])
        
        # Si la distancia es menor que un umbral, consideramos que hay un gesto de "pinch"
        return distance < 20

    def is_fist_closed(hand_landmarks):
        # Comprueba si las puntas de los dedos están por debajo de las articulaciones
        landmarks = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip, dip in landmarks:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
                return False
        return True

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Voltea la imagen horizontalmente para una vista de espejo
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Dibuja los puntos y conexiones de la mano en la imagen
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Comprueba si el gesto de "pinch" está activo
                if is_pinch(hand_landmarks, image.shape):
                    # Obtén las coordenadas del punto 8 (dedo índice)
                    index_finger_tip = hand_landmarks.landmark[8]
                    h, w, _ = image.shape
                    cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    
                    # Almacena las coordenadas para dibujar
                    points.append((cx, cy))

                # Comprueba si el puño está cerrado
                if is_fist_closed(hand_landmarks):
                    points = []  # Borra los puntos si la mano está cerrada

        # Dibuja las líneas entre los puntos almacenados para suavizar el trazo
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], drawing_color, drawing_thickness)

        # Muestra la imagen
        cv2.imshow('Hand Writing', image)
        key = cv2.waitKey(5)
        
        if key & 0xFF == 27:  # Presiona ESC para salir
            break
        elif key & 0xFF == ord('s'):  # Presiona 's' para guardar la imagen
            # Guarda la imagen
            drawing = np.zeros_like(image)
            for i in range(1, len(points)):
                cv2.line(drawing, points[i-1], points[i], drawing_color, drawing_thickness)
            cv2.imwrite('hand_drawing.png', drawing)
            print('Drawing saved as hand_drawing.png')
        elif key & 0xFF == ord('c'):  # Presiona 'c' para cambiar el color del trazo
            drawing_color = (255, 0, 0)  # Cambia a rojo
        elif key & 0xFF == ord('t'):  # Presiona 't' para cambiar el grosor del trazo
            drawing_thickness = 5  # Aumenta el grosor del trazo

    cap.release()
    cv2.destroyAllWindows()
