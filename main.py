import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.screen_width, self.screen_height = pyautogui.size()
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.prev_right_hand_center = None
        self.cursor_speed_multiplier = 4
        
        self.gesture_states = {
            'right_primary_click': False,
            'left_secondary_click': False,
            'scroll_up': False,
            'scroll_down': False
        }
        
        self.frame_width = None
        self.frame_height = None
    
    def calculate_distance(self, point1, point2):
        return math.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2
        )
    
    def get_hand_center(self, hand_landmarks):
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
        return np.mean(x_coords), np.mean(y_coords)
    
    def detect_gestures(self, hand_landmarks, hand_label):
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        pinch_threshold = 0.04
        
        gestures = []
        
        if (hand_label.lower() == 'right' and 
            thumb_index_dist < pinch_threshold):
            gestures.append('right_primary_click')
        
        if (hand_label.lower() == 'left' and 
            thumb_index_dist < pinch_threshold):
            gestures.append('left_secondary_click')
        
        if hand_label.lower() == 'left':
            if (thumb_tip.y > index_tip.y and 
                thumb_tip.y > middle_tip.y and 
                thumb_tip.y > ring_tip.y):
                gestures.append('scroll_down')
            
            if (thumb_tip.y < index_tip.y and 
                thumb_tip.y < middle_tip.y and 
                thumb_tip.y < ring_tip.y):
                gestures.append('scroll_up')
        
        return gestures
    
    def calculate_cursor_movement(self, current_center):
        if self.prev_right_hand_center is None:
            return 0, 0
        
        dx = (current_center[0] - self.prev_right_hand_center[0]) * self.frame_width
        dy = (current_center[1] - self.prev_right_hand_center[1]) * self.frame_height
        
        dx *= self.cursor_speed_multiplier
        dy *= self.cursor_speed_multiplier
        
        return dx, dy
    
    def perform_gestures(self, gestures):
        if 'right_primary_click' in gestures and not self.gesture_states['right_primary_click']:
            pyautogui.click()
            self.gesture_states['right_primary_click'] = True
        elif 'right_primary_click' not in gestures:
            self.gesture_states['right_primary_click'] = False
        
        if 'left_secondary_click' in gestures and not self.gesture_states['left_secondary_click']:
            pyautogui.rightClick()
            self.gesture_states['left_secondary_click'] = True
        elif 'left_secondary_click' not in gestures:
            self.gesture_states['left_secondary_click'] = False
        
        if 'scroll_down' in gestures and not self.gesture_states['scroll_down']:
            pyautogui.scroll(-20)
            self.gesture_states['scroll_down'] = True
        elif 'scroll_down' not in gestures:
            self.gesture_states['scroll_down'] = False
        
        if 'scroll_up' in gestures and not self.gesture_states['scroll_up']:
            pyautogui.scroll(20)
            self.gesture_states['scroll_up'] = True
        elif 'scroll_up' not in gestures:
            self.gesture_states['scroll_up'] = False
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            self.frame_height, self.frame_width, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(rgb_frame)
            
            right_hand = None
            left_hand = None
            
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_classification = results.multi_handedness[hand_no].classification[0]
                    hand_label = hand_classification.label.lower()
                    
                    if hand_label == 'right':
                        right_hand = (hand_landmarks, hand_label)
                    else:
                        left_hand = (hand_landmarks, hand_label)
                
                if right_hand and left_hand:
                    right_landmarks, right_label = right_hand
                    
                    right_center = self.get_hand_center(right_landmarks)
                    
                    if self.prev_right_hand_center is not None:
                        dx, dy = self.calculate_cursor_movement(right_center)
                        
                        current_x, current_y = pyautogui.position()
                        
                        new_x = max(0, min(self.screen_width, current_x + dx))
                        new_y = max(0, min(self.screen_height, current_y + dy))
                        
                        pyautogui.moveTo(new_x, new_y)
                    
                    self.prev_right_hand_center = right_center
                
                all_gestures = []
                if right_hand:
                    right_landmarks, right_label = right_hand
                    all_gestures.extend(self.detect_gestures(right_landmarks, right_label))
                
                if left_hand:
                    left_landmarks, left_label = left_hand
                    all_gestures.extend(self.detect_gestures(left_landmarks, left_label))
                
                if all_gestures:
                    self.perform_gestures(all_gestures)
                else:
                    self.prev_right_hand_center = None
            
            if results.multi_hand_landmarks:
                for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_classification = results.multi_handedness[hand_no].classification[0]
                    hand_label = hand_classification.label
                    
                    h, w, _ = frame.shape
                    
                    connections = self.mp_hands.HAND_CONNECTIONS
                    for connection in connections:
                        start_point = hand_landmarks.landmark[connection[0]]
                        end_point = hand_landmarks.landmark[connection[1]]
                        
                        cv2.line(
                            frame, 
                            (int(start_point.x * w), int(start_point.y * h)), 
                            (int(end_point.x * w), int(end_point.y * h)), 
                            (0, 0, 0),
                            2
                        )
                    
                    finger_tips = [4, 8, 12, 16, 20]
                    
                    for i, tip_idx in enumerate(finger_tips, start=1):
                        landmark = hand_landmarks.landmark[tip_idx]
                        
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        
                        cv2.circle(
                            frame, 
                            (cx, cy), 
                            5,
                            (0, 0, 255),
                            -1
                        )
                        
                        cv2.putText(
                            frame, 
                            str(i), 
                            (cx+10, cy+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0),
                            2
                        )
                    
                    x_pos = 10 if hand_label == 'Left' else frame.shape[1] - 150
                    cv2.putText(
                        frame, 
                        f'Hand: {hand_label}', 
                        (x_pos, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0),
                        2
                    )
            
            cv2.imshow('Hand Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()