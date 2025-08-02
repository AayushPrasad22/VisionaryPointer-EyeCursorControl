#importing necessary modules
import cv2                                           # for accessing webcam and image processing
import mediapipe as mp                               # for detecting face landmarks
import pyautogui                                     # for controlling the mouse

# Start capturing video from the default webcam
cam = cv2.VideoCapture(0)

# Initialize MediaPipe FaceMesh with landmark refinement
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen width and height for mapping face coordinates to screen
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()  # Read a frame from the webcam
    frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    output = face_mesh.process(rgb_frame)  # Process frame to detect face landmarks
    landmark_points = output.multi_face_landmarks  # Get landmarks if detected
    frame_h, frame_w, _ = frame.shape  # Get the dimensions of the frame

    if landmark_points:
        landmarks = landmark_points[0].landmark  # Get the landmarks of the first detected face

        # Loop over landmarks 474 to 477 (right eye region)
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)  # Convert normalized x to pixel value
            y = int(landmark.y * frame_h)  # Convert normalized y to pixel value
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw green circle on screen
            if id == 1:  # Track movement using the second point (id 1)
                screen_x = screen_w * landmark.x  # Map x to screen coordinates
                screen_y = screen_h * landmark.y  # Map y to screen coordinates
                pyautogui.moveTo(screen_x, screen_y)  # Move the mouse to mapped coordinates

        # Get eye blink landmarks (for left eye)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Draw yellow circles for eye blink tracking

        # Check if the vertical distance between the eyelid points is small (blink detected)
        if (left[0].y - left[1].y) < 0.007:
            pyautogui.click()  # Trigger a mouse click
            pyautogui.sleep(0.2)  # Wait for 0.2 seconds to prevent multiple clicks

    cv2.imshow('Eye Controlled Mouse', frame)  # Show the video with overlays

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cam.release()
cv2.destroyAllWindows()
