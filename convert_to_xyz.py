import mediapipe as mp
import cv2

# model prep
model_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5,  min_tracking_confidence=0.5)

# camp prep
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
cap.set(cv2.CAP_PROP_FPS, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.flip(frame, 1)  # Mirror display

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    image.flags.writeable = False
    results = hands.process(rgb_frame)
    image.flags.writeable = True

    # Extract hand landmarks
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Get the top of the pointer finger (index 8)
            x, y, z = landmarks.landmark[8].x, landmarks.landmark[8].y, landmarks.landmark[8].z

            # Normalize x and y coordinates to [0.0, 1.0] by image width and height
            image_width, image_height = frame.shape[1], frame.shape[0]
            x_normalized, y_normalized = x / image_width, y / image_height

            # Draw a blue dot at the fingertip position
            cv2.circle(frame, (x_normalized, y_normalized), radius=5, color=(255, 0, 0), thickness=100)

            # The z coordinate represents landmark depth (wrist depth is the reference)
            # You can use this information to calculate real-world coordinates

            # Print the results
            print(f"Normalized X: {x_normalized:.2f}, Y: {y_normalized:.2f}, Z: {z:.2f}")

    # Display the frame (you can add visualization here)
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
