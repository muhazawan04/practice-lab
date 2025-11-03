import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Canvas for drawing
canvas = None

# Drawing color and thickness
draw_color = (0, 0, 255)  # Red
thickness = 3
min_movement = 5

# To store previous finger position
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Get index fingertip and thumb tip
        index_finger_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]

        # Convert to pixel coords
        x1, y1 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        x2, y2 = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # Compute distance between them
        distance = math.hypot(x2 - x1, y2 - y1)
        if key == ord('r'):
                draw_color = (0, 0, 255)
        elif key == ord('g'):
            draw_color = (0, 255, 0)
        elif key == ord('b'):
            draw_color = (255, 0, 0)

        # If they are close enough → draw mode
        if distance < 20:
            cv2.circle(frame, (x1, y1), 10, draw_color, -1)
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x1, y1

            movement = math.hypot(x1 - prev_x, y1 - prev_y)
            if movement > min_movement:
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), draw_color, thickness)
                prev_x, prev_y = x1, y1

        else:
            prev_x, prev_y = 0, 0  # reset

    # Merge drawings with frame
    frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    cv2.imshow("Virtual Pen", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('c'):  # 'c' to clear screen
        canvas = np.zeros((h, w, 3), np.uint8)

cap.release()
cv2.destroyAllWindows()
