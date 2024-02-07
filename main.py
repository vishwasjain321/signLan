import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4
index_finger_tip = 8
middle_finger_tip = 12

like_img = cv2.imread("images/like.jpg")
like_img = cv2.resize(like_img, (200, 180))

dislike_img = cv2.imread("images/dislike.jpg")
dislike_img = cv2.resize(dislike_img, (200, 180))

victory_img = cv2.imread("images/victory.jpg")
victory_img = cv2.resize(victory_img, (200, 180))

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = [lm for lm in hand_landmark.landmark]
            finger_fold_status = []

            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                print(id, ":", x, y)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            # Your gesture recognition code remains unchanged

            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y:
                    print("Good")
                    cv2.putText(img, "GOOD", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    h, w, c = like_img.shape
                    img[35:h + 35, 30:w + 30] = like_img
                elif lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y:
                    cv2.putText(img, "BAD", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("Bad")
                    h, w, c = dislike_img.shape
                    img[35:h + 35, 30:w + 30] = dislike_img

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2)
                                   )

    cv2.imshow("Hand Gesture Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
