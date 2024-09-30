import cv2
import face_recognition
import numpy as np

# Initialize known face encodings and names
known_face_encodings = []
known_face_names = []

# Known faces (ensure that this image file exists in the same folder)
known_images = {"Kacper Wrona": "kacper_wrona.jpeg"}

# Load known face images and get encodings
for name, image_path in known_images.items():
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image)

        # Ensure face locations were found
        if face_locations:
            # Extract face encodings for the detected face(s)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(name)
                print(f"Loaded face encoding for {name}.")
            else:
                print(f"No face encodings found for {image_path}")
        else:
            print(f"No face found in {image_path}")
    except FileNotFoundError:
        print(f"Image file {image_path} not found.")
        continue

# Start video capture
video_capture = cv2.VideoCapture(0)

# Check if video capture is opened
if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

# Set lower resolution for performance
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # Check if frame capture was successful
    if not ret:
        print("Failed to capture video frame.")
        continue

    # Convert the image from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    print(f"Detected {len(face_locations)} faces in the current frame.")

    # Check if any faces are found
    if face_locations:  # Check if any faces are found
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        # Draw boxes around the detected faces
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()