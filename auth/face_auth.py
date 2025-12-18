import cv2
import numpy as np
import time
import os


class FaceAuthenticator:
    """
    OpenCV DNN-based Multi-Sample Owner Authentication
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")
        owner_dir = os.path.join(base_dir, "owner_face")

        # Load face detector
        self.face_net = cv2.dnn.readNetFromCaffe(
            os.path.join(model_dir, "deploy.prototxt"),
            os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        )

        # Load embedding model
        self.embedder = cv2.dnn.readNetFromTorch(
            os.path.join(model_dir, "nn4.small2.v1.t7")
        )

        # Load all owner embeddings
        self.owner_embeddings = self._load_owner_embeddings(owner_dir)

        if not self.owner_embeddings:
            raise ValueError("No valid owner faces found.")

    # -------------------------------
    # Load and encode owner faces
    # -------------------------------
    def _load_owner_embeddings(self, owner_dir):
        embeddings = []

        for file in os.listdir(owner_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(owner_dir, file)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                face = self._extract_face(image)
                if face is None:
                    continue

                emb = self._get_embedding(face)
                embeddings.append(emb)

        return embeddings

    # -------------------------------
    # Face detection
    # -------------------------------
    def _extract_face(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        if detections.shape[2] == 0:
            return None

        confidence = detections[0, 0, 0, 2]
        if confidence < 0.6:
            return None

        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        return frame[y1:y2, x1:x2]

    # -------------------------------
    # Generate embedding
    # -------------------------------
    def _get_embedding(self, face):
        face_blob = cv2.dnn.blobFromImage(
            face, 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False
        )
        self.embedder.setInput(face_blob)
        return self.embedder.forward().flatten()

    # -------------------------------
    # Authenticate (LIVE)
    # -------------------------------
    def authenticate(self, timeout=8, threshold=0.75):
        cap = cv2.VideoCapture(0)
        start = time.time()

        access_mode = "GUEST"

        while time.time() - start < timeout:
            ret, frame = cap.read()
            if not ret:
                continue

            face = self._extract_face(frame)
            if face is None:
                continue

            live_embedding = self._get_embedding(face)

            # Compare against all owner embeddings
            for owner_emb in self.owner_embeddings:
                distance = np.linalg.norm(live_embedding - owner_emb)
                if distance < threshold:
                    access_mode = "OWNER"
                    break

            if access_mode == "OWNER":
                break

        cap.release()
        cv2.destroyAllWindows()
        return access_mode
