import cv2
import face_recognition
import numpy as np

known_face_encodings = []
known_face_names = []

known_images = {"Kacper Wrona": "kacper_wrona.jpeg", "Tomasz Jankowy": "tomasz_jankowy.jpeg", "Alicja Burdzy": "alicja_burdzy.jpg"}

for name, image_path in known_images.items():
    try:
        image = face_recognition.load_image_file(image_path)

        face_locations = face_recognition.face_locations(image)

        if face_locations:
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

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = video_capture.read()
    
    if not ret:
        print("Failed to capture video frame.")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    print(f"Detected {len(face_locations)} faces in the current frame.")

    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()