import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh


class CVEngine:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
        )
        self.eyes_closed_frames = 0

    def eye_aspect_ratio(self, eye_points):
        # Compute EAR
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def extract_eye(self, landmarks, indices, w, h):
        return np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
            dtype=np.float32,
        )

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        drowsiness = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark

            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]

            left_eye = self.extract_eye(landmarks, left_eye_idx, w, h)
            right_eye = self.extract_eye(landmarks, right_eye_idx, w, h)

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # EAR threshold
            if ear < 0.22:
                self.eyes_closed_frames += 1
            else:
                self.eyes_closed_frames = 0

            # Normalize drowsiness score
            drowsiness = min(1.0, self.eyes_closed_frames / 10.0)

        return {
            "cv_features": {
                "face_detected": face_detected,
                "drowsiness": round(drowsiness, 4),
                "eyes_closed_frames": int(self.eyes_closed_frames),
            }
        }
