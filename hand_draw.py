import os
from time import sleep

import cv2
import mediapipe as mp
import numpy as np


WIDTH = 1280
HEIGHT = 720
COLORS = {
    'BLUE': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'RED': (0, 0, 255),
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
}
BLANCK_BOARD = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

image_board = BLANCK_BOARD.copy()
board_x = 0
board_y = 0
pencil_color = (255, 0, 0)


def close_n_exit(video):
    video.release()
    cv2.destroyAllWindows()


def locate_hands(image):
    rgb_mode = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_mode)
    hands_located = []

    if result.multi_hand_landmarks:

        for side, hand_markers in zip(result.multi_handedness, result.multi_hand_landmarks):
            hands_located.append({
                'coordinates': [(int(marker.x * WIDTH), int(marker.y * HEIGHT), int(marker.z * WIDTH)) for marker in hand_markers.landmark],
                'side': side.classification[0].label
            })
            mp_draw.draw_landmarks(image,
                                    hand_markers,
                                    mp_hands.HAND_CONNECTIONS)

    return image, hands_located


def fingers_raised(hand):
    return [True if hand['coordinates'][pointer][1] < hand['coordinates'][pointer - 2][1] else False for pointer in [8, 12, 16, 20]]


def set_pencil_color(hand):
    if sum(hand) == 0:
        return COLORS['WHITE']
    elif sum(hand) == 1:
        return COLORS['RED']
    elif sum(hand) == 2:
        return COLORS['GREEN']
    elif sum(hand) == 3:
        return COLORS['BLUE']
    else:
        return COLORS['BLACK']


def create_draw_board(image, hands):
    global image_board
    global board_x
    global board_y
    fingers_hand_a = fingers_raised(hands[0])
    fingers_hand_b = fingers_raised(hands[1])

    if sum(fingers_hand_b) > 4:
        image_board = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255
    else:
        pencil_color = set_pencil_color(fingers_hand_b)
    
    finger_x, finger_y, finger_z = hands[0]['coordinates'][8]

    pencil_size = int(abs(finger_z)) // 4

    cv2.circle(
        image,
        (finger_x, finger_y),
        pencil_size,
        pencil_color,
        cv2.FILLED
    )

    if fingers_hand_a == [True, False, False, False]:
        
        if board_x == 0 and board_y == 0:
            board_x, board_y = finger_x, finger_y
        
        cv2.line(
            image_board,
            (board_x, board_y),
            (finger_x, finger_y),
            pencil_color,
            pencil_size
        )

        board_x, board_y = finger_x, finger_y
    
    else:

        board_x, board_y = 0, 0
    
    return cv2.addWeighted(image, 1, image_board, .2, 0)

def main():
    try:

        video_input = cv2.VideoCapture(0)
        video_input.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        video_input.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        while video_input.isOpened():

            _ok_render, image = video_input.read()
            image = cv2.flip(image, 1)

            image, hands_located = locate_hands(image)

            if len(hands_located) == 2:
                image = create_draw_board(image, hands_located)

            cv2.imshow('You', image)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                if image_board.sum() < BLANCK_BOARD.sum():
                    cv2.imwrite('board.png', image_board)
                close_n_exit(video_input)

    except Exception as err:
        print(err)

if __name__ == '__main__':
    main()