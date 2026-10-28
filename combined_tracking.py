import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# =========================================================
# MEDIAPIPE HAND LANDMARKER
# =========================================================

hand_base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    num_hands=1
)

hand_detector = vision.HandLandmarker.create_from_options(
    hand_options
)


# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

face_base_options = python.BaseOptions(
    model_asset_path="models/face_landmarker.task"
)

face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    num_faces=1
)

face_detector = vision.FaceLandmarker.create_from_options(
    face_options
)


# =========================================================
# HAND GESTURE DETECTION
# =========================================================

def fingers_up(hand):
    """
    Returns:

    [thumb, index, middle, ring, pinky]

    True  = finger extended
    False = finger folded
    """

    fingers = []

    # -----------------------------------------------------
    # THUMB
    # -----------------------------------------------------

    if hand[4].x < hand[3].x:
        fingers.append(True)
    else:
        fingers.append(False)


    # -----------------------------------------------------
    # INDEX
    # -----------------------------------------------------

    fingers.append(
        hand[8].y < hand[6].y
    )


    # -----------------------------------------------------
    # MIDDLE
    # -----------------------------------------------------

    fingers.append(
        hand[12].y < hand[10].y
    )


    # -----------------------------------------------------
    # RING
    # -----------------------------------------------------

    fingers.append(
        hand[16].y < hand[14].y
    )


    # -----------------------------------------------------
    # PINKY
    # -----------------------------------------------------

    fingers.append(
        hand[20].y < hand[18].y
    )


    return fingers


# =========================================================
# GESTURE CLASSIFIER
# =========================================================

def detect_gesture(hand):

    fingers = fingers_up(hand)

    thumb = fingers[0]
    index = fingers[1]
    middle = fingers[2]
    ring = fingers[3]
    pinky = fingers[4]


    # -----------------------------------------------------
    # OPEN PALM
    # -----------------------------------------------------

    if thumb and index and middle and ring and pinky:
        return "OPEN PALM"


    # -----------------------------------------------------
    # CLOSED PALM / FIST
    # -----------------------------------------------------

    if not thumb and not index and not middle and not ring and not pinky:
        return "CLOSED PALM"


    # -----------------------------------------------------
    # ONE FINGER
    # -----------------------------------------------------

    if index and not middle and not ring and not pinky:
        return "ONE FINGER"


    # -----------------------------------------------------
    # PEACE SIGN
    # -----------------------------------------------------

    if index and middle and not ring and not pinky:
        return "PEACE"


    # -----------------------------------------------------
    # THREE FINGERS
    # -----------------------------------------------------

    if index and middle and ring and not pinky:
        return "THREE FINGERS"


    # -----------------------------------------------------
    # THUMBS UP / DOWN
    # -----------------------------------------------------

    if thumb and not index and not middle and not ring and not pinky:

        if hand[4].y < hand[3].y:
            return "THUMBS UP"

        elif hand[4].y > hand[3].y:
            return "THUMBS DOWN"


    return "UNKNOWN"


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()


print("Face + Hand tracking started")
print("Press Q to quit")


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera")
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    height, width, _ = frame.shape


    # =====================================================
    # CONVERT FRAME
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # =====================================================
    # RUN FACE DETECTION
    # =====================================================

    face_result = face_detector.detect(mp_image)


    # =====================================================
    # RUN HAND DETECTION
    # =====================================================

    hand_result = hand_detector.detect(mp_image)


    # =====================================================
    # FACE TRACKING
    # =====================================================

    if face_result.face_landmarks:

        face = face_result.face_landmarks[0]


        # Find face bounding box

        xs = [landmark.x for landmark in face]
        ys = [landmark.y for landmark in face]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)


        # Face center

        face_x = (min_x + max_x) / 2
        face_y = (min_y + max_y) / 2


        # Convert to pixels

        face_pixel_x = int(
            face_x * width
        )

        face_pixel_y = int(
            face_y * height
        )


        # Draw face center

        cv2.circle(
            frame,
            (face_pixel_x, face_pixel_y),
            10,
            (0, 0, 255),
            -1
        )


        # Draw face landmarks

        for landmark in face:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )


        # Face position

        cv2.putText(
            frame,
            f"Face X: {face_x:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Face Y: {face_y:.2f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # =====================================================
    # HAND TRACKING
    # =====================================================

    if hand_result.hand_landmarks:

        hand = hand_result.hand_landmarks[0]


        # Detect gesture

        gesture = detect_gesture(hand)



        # =====================================================
        # DRAW HAND SKELETON
        # =====================================================

        # Draw connections first
        for start, end in HAND_CONNECTIONS:

            x1 = int(hand[start].x * width)
            y1 = int(hand[start].y * height)

            x2 = int(hand[end].x * width)
            y2 = int(hand[end].y * height)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )


        # Draw landmarks on top
        for landmark in hand:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (255, 0, 0),
                -1
            )


        # -------------------------------------------------
        # Hand center
        # -------------------------------------------------

        hand_x = int(
            sum(l.x for l in hand)
            / len(hand)
            * width
        )

        hand_y = int(
            sum(l.y for l in hand)
            / len(hand)
            * height
        )


        # -------------------------------------------------
        # Display gesture
        # -------------------------------------------------

        cv2.putText(
            frame,
            gesture,
            (30, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "Face + Hand Tracking",
        frame
    )


    # =====================================================
    # QUIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()

hand_detector.close()
face_detector.close()