import os
import cv2
import string
import mediapipe as mp
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(script_dir, 'DataSet')

# Creating the directory Structure
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(os.path.join(DATA_DIR, "testingData")):
    os.makedirs(os.path.join(DATA_DIR, "testingData"))

number_of_classes = 26  # A to Z
dataset_size = 50
current_class_index = 0

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
 
# Define colors for each finger
finger_colors = {
    "Thumb": (0, 0, 255),    # Red
    "Index": (0, 255, 0),    # Green
    "Middle": (255, 0, 0),   # Blue
    "Ring": (128, 0, 128),   # Purple
    "Pinky": (255, 255, 255)   # White
}

labels = list(string.ascii_uppercase)

def imageCapture(ret,frame,results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            # Determine if the hand is the right hand or left hand
            is_right_hand = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Draw hand type
            hand_type_text = "Right Hand" if is_right_hand else "Left Hand"
            cv2.putText(frame, hand_type_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Draw landmarks with different colors for each finger
            for finger, color in finger_colors.items():
                for connection in mp_hands.HAND_CONNECTIONS:
                    point1, point2 = connection
                    try:
                        index1 = mp_hands.HandLandmark[point1].value
                        index2 = mp_hands.HandLandmark[point2].value
                        if index1 < len(hand_landmarks.landmark) and index2 < len(hand_landmarks.landmark):
                            x1, y1 = int(hand_landmarks.landmark[index1].x * frame.shape[1]), int(hand_landmarks.landmark[index1].y * frame.shape[0])
                            x2, y2 = int(hand_landmarks.landmark[index2].x * frame.shape[1]), int(hand_landmarks.landmark[index2].y * frame.shape[0])
                            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                    except KeyError:
                        pass
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Assign colors to individual finger landmarks
            for finger, color in finger_colors.items():
                if finger == "Thumb":
                    # Thumb landmarks indices
                    indices = [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP]
                elif finger == "Index":
                    # Index finger landmarks indices
                    indices = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]
                elif finger == "Middle":
                    # Middle finger landmarks indices
                    indices = [mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                elif finger == "Ring":
                    # Ring finger landmarks indices
                    indices = [mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP]
                elif finger == "Pinky":
                    # Pinky finger landmarks indices
                    indices = [mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP]

                # Draw lines between the landmarks with the specified color
                for index in indices:
                    landmark = hand_landmarks.landmark[index]
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, color, -1)  # Draw a filled circle for the landmark

while current_class_index < number_of_classes:
    current_class = labels[current_class_index]

    class_folder = os.path.join(DATA_DIR, "testingData", current_class)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    print('Collecting data for class {}'.format(current_class))

    cap = cv2.VideoCapture(0)  # Move inside the loop

    # Displaying instructions until 'q' or ESC is pressed
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        imageCapture(ret,frame,results)
        cv2.putText(frame, 'Press "Q" to capture or ESC to exit!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)  # Changed color to red and position to top-left corner
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Capturing images
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        imageCapture(ret,frame,results)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_folder, '{}.jpg'.format(counter)), frame)
        counter += 1

    cap.release()
    cv2.destroyAllWindows()

    current_class_index += 1