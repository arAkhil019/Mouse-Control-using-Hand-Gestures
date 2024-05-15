import cv2
import mediapipe as mp
import v3b
import pyautogui
from pynput.mouse import Button, Controller

#setting mouse
mouse = Controller()
#getting screen sizes
screen_width, screen_height = pyautogui.size()

# setting up the model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


# to find tips
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

#to move mouse using pyautogui
def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x,y)

#function to left click
def is_left_click(landmarks_list,thumb_index_dist):
    index_fing_angle = v3b.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])
    middle_fing_angle = v3b.get_angle(landmarks_list[9],landmarks_list[10],landmarks_list[12])

    return (index_fing_angle < 50 and middle_fing_angle > 90 and thumb_index_dist > 50)
#for right click
def is_right_click(landmarks_list,thumb_index_dist):
    index_fing_angle = v3b.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])
    middle_fing_angle = v3b.get_angle(landmarks_list[9],landmarks_list[10],landmarks_list[12])

    return (index_fing_angle > 90 and middle_fing_angle < 50 and thumb_index_dist > 50)

# detect gestures function
def detect_gestures(frame, landmarks_list, processed):
    if len(landmarks_list) >= 21:
        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = v3b.get_distance([landmarks_list[4], landmarks_list[5]])
        angle_of_index = v3b.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])

        #to move mouse
        if thumb_index_dist<50 and angle_of_index >90:
            move_mouse(index_finger_tip)
        #to left click
        elif is_left_click(landmarks_list,thumb_index_dist):
            mouse.click(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, 'Left Clicked',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        #to right click
        elif is_right_click(landmarks_list,thumb_index_dist):
            mouse.click(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, 'Right Clicked',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

# main driver function
def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # as opencv default is BGR
            processed = hands.process(frameRGB)

            landmarks_list = []

            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x, lm.y))

            detect_gestures(frame, landmarks_list, processed)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
