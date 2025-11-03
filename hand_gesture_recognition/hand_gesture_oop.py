import cv2
import mediapipe as mp
import numpy as np
import math

class VirtualWhiteboard:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Drawing setup
        self.canvas = None
        self.draw_color = (0, 0, 255)  # default red
        self.thickness = 5
        self.prev_x, self.prev_y = 0, 0
        self.min_movement = 5

        # Toolbar setup
        self.toolbar_height = 100
        self.button_size = (60, 50)
        self.spacing = 20
        self.colors = [
            ((0, 0, 255), "Red"),
            ((0, 255, 0), "Green"),
            ((255, 0, 0), "Blue"),
            ((0, 0, 0), "Eraser"),
        ]
        self.active_color_index = 0

    def draw_toolbar(self, frame):
        """Draw color palette and brush controls (centered on top)."""
        h, w, _ = frame.shape
        num_buttons = len(self.colors) + 2  # colors + (+, -)
        total_width = num_buttons * (self.button_size[0] + self.spacing) - self.spacing
        start_x = (w - total_width) // 2  # Centered starting position

        x_offset = start_x
        for i, (color, name) in enumerate(self.colors):
            border = 4 if i == self.active_color_index else 2
            cv2.rectangle(frame, (x_offset, 10),
                          (x_offset + self.button_size[0], 10 + self.button_size[1]),
                          color, -1)
            cv2.rectangle(frame, (x_offset, 10),
                          (x_offset + self.button_size[0], 10 + self.button_size[1]),
                          (255, 255, 255), border)
            x_offset += self.button_size[0] + self.spacing

        # + button
        cv2.rectangle(frame, (x_offset, 10),
                      (x_offset + self.button_size[0], 10 + self.button_size[1]),
                      (50, 50, 50), -1)
        cv2.putText(frame, '+', (x_offset + 18, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        x_offset += self.button_size[0] + self.spacing

        # - button
        cv2.rectangle(frame, (x_offset, 10),
                      (x_offset + self.button_size[0], 10 + self.button_size[1]),
                      (50, 50, 50), -1)
        cv2.putText(frame, '-', (x_offset + 22, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    def check_toolbar_selection(self, x, y, frame_width):
        """Handles index finger hovering over toolbar for color/brush selection."""
        if y > self.toolbar_height:
            return

        num_buttons = len(self.colors) + 2
        total_width = num_buttons * (self.button_size[0] + self.spacing) - self.spacing
        start_x = (frame_width - total_width) // 2
        x_offset = start_x

        # Color buttons
        for i, (color, name) in enumerate(self.colors):
            if x_offset < x < x_offset + self.button_size[0] and 10 < y < 10 + self.button_size[1]:
                self.draw_color = color
                self.active_color_index = i
                print(f"Selected color: {name}")
                return
            x_offset += self.button_size[0] + self.spacing

        # + button
        if x_offset < x < x_offset + self.button_size[0] and 10 < y < 10 + self.button_size[1]:
            self.thickness += 1
            print(f"Brush size increased: {self.thickness}")
            return
        x_offset += self.button_size[0] + self.spacing

        # - button
        if x_offset < x < x_offset + self.button_size[0] and 10 < y < 10 + self.button_size[1]:
            self.thickness = max(1, self.thickness - 1)
            print(f"Brush size decreased: {self.thickness}")
            return
    def get_finger_distance(self, tip, base):
        distance = math.hypot(tip.x - base.x, tip.y - base.y)
        return distance


    def process_hand(self, frame):
        #"""Handles hand tracking and drawing."""
        h, w, _ = frame.shape
        if self.canvas is None:
            self.canvas = np.zeros((h, w, 3), np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Get fingertips
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            thumb_tip = hand_landmarks.landmark[4]
            
            # Get palm base points
            wrist = hand_landmarks.landmark[0]
            palm_center = hand_landmarks.landmark[9]  # Middle finger MCP joint
            palm_length = self.get_finger_distance(palm_center, wrist)
            print(f"Palm length: {palm_length}")    
            
            # Check if fingers are closed using normalized distance thresholds
            thumb_dist = self.get_finger_distance(thumb_tip, wrist)
            ring_dist = self.get_finger_distance(ring_tip, palm_center)
            pinky_dist = self.get_finger_distance(pinky_tip, palm_center)

            # Define distance thresholds (normalized)
            CLOSED_THRESHOLD = 0.9 * palm_length  # 15% of palm length
            print(f"Thumb dist: {thumb_dist}, threshold: {CLOSED_THRESHOLD}")
            is_ring_closed = ring_dist < CLOSED_THRESHOLD
            is_pinky_closed = pinky_dist < CLOSED_THRESHOLD
            is_thumb_closed = thumb_dist < CLOSED_THRESHOLD

            x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2, y2 = int(middle_tip.x * w), int(middle_tip.y * h)
            distance = self.get_finger_distance(index_tip, middle_tip) / palm_length  # Normalized
            print(f"Index-Middle normalized distance: {distance}")

            # Drawing logic
            if y1 < self.toolbar_height:
                self.check_toolbar_selection(x1, y1, w)
                self.prev_x, self.prev_y = 0, 0
            elif distance < 0.25 and is_ring_closed and is_pinky_closed and is_thumb_closed:
                draw_x = (x1 + x2) // 2
                draw_y = (y1 + y2) // 2
                
                circle_radius = max(3, self.thickness)
                cv2.circle(frame, (draw_x, draw_y), circle_radius, self.draw_color, -1)

                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = draw_x, draw_y

                movement = math.hypot(draw_x - self.prev_x, draw_y - self.prev_y)
                if movement > self.min_movement:
                    cv2.line(self.canvas, (self.prev_x, self.prev_y),
                            (draw_x, draw_y), self.draw_color, self.thickness)
                    self.prev_x, self.prev_y = draw_x, draw_y
            else:
                self.prev_x, self.prev_y = 0, 0

        return frame


    

    def run(self):
        """Main loop."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            self.draw_toolbar(frame)
            frame = self.process_hand(frame)
            frame = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)

            cv2.putText(frame, f'Brush: {self.thickness}', (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Virtual Whiteboard", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == ord('c'):
                self.canvas = np.zeros((h, w, 3), np.uint8)
                print("Canvas cleared")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    board = VirtualWhiteboard()
    board.run()
