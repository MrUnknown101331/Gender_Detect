from tensorflow.keras.models import model_from_json
import numpy as np
import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)
mp_face_detection = mp.solutions.face_detection
class_labels = ["Male", "Female"]

# load json and create model
json_file = open('models/GenderModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/GenderModel.weights.h5")
print("Loaded model from disk")

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not success:
        break

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        annotated_image = frame.copy()
        if results.detections:
            print("Number of people detected ", len(results.detections), "\n\n")
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                box_width = int(bbox.width * w)
                box_height = int(bbox.height * h)

                # Ensure the bounding box coordinates are within the frame dimensions
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_min + box_width)
                y_max = min(h, y_min + box_height)

                x_start, y_start = x_min, y_min
                x_end, y_end = x_min + box_width, y_min + box_height
                annotated_image = cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                face = frame[y_min:y_max, x_min:x_max]
                resized_face = cv2.resize(face, (64, 64))

                resized_face = resized_face / 255
                expanded_face = np.expand_dims(resized_face, axis=0)
                prediction = loaded_model.predict(expanded_face, batch_size=None, steps=1)  # gives all class prob.
                predicted_class = class_labels[np.argmax(prediction)]

                cv2.putText(frame, str(predicted_class), (x_start, y_start), cv2.FONT_HERSHEY_DUPLEX,
                            1,
                            (255, 5, 255),
                            2)
            cv2.putText(frame, "Number of people detected " + str(len(results.detections)), (20, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 5, 255), 2)

    cv2.imshow('Video', annotated_image)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
