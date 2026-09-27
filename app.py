import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="centered"
)

st.title("🚗 Driver Drowsiness Detection System")
st.write("Hybrid System using ResNet50 + FaceMesh 468 + EAR + MAR")


@st.cache_resource
def load_models():
    eye_model = load_model("MRL_resnet50_best.h5")
    mouth_model = load_model("Yawn_resnet50_best.h5")
    return eye_model, mouth_model


eye_model, mouth_model = load_models()

st.success("✅ Models Loaded Successfully")


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)


def EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    return (A + B) / (2.0 * C)


def MAR(mouth):
    A = np.linalg.norm(mouth[1] - mouth[7])
    B = np.linalg.norm(mouth[2] - mouth[6])
    C = np.linalg.norm(mouth[3] - mouth[5])
    D = np.linalg.norm(mouth[0] - mouth[4])

    return (A + B + C) / (2.0 * D)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]


uploaded_file = st.file_uploader(
    "Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = image.shape

            left_eye = np.array([
                [
                    int(face_landmarks.landmark[i].x * w),
                    int(face_landmarks.landmark[i].y * h)
                ]
                for i in LEFT_EYE
            ])

            right_eye = np.array([
                [
                    int(face_landmarks.landmark[i].x * w),
                    int(face_landmarks.landmark[i].y * h)
                ]
                for i in RIGHT_EYE
            ])

            mouth = np.array([
                [
                    int(face_landmarks.landmark[i].x * w),
                    int(face_landmarks.landmark[i].y * h)
                ]
                for i in MOUTH
            ])

            leftEAR = EAR(left_eye)
            rightEAR = EAR(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            mar = MAR(mouth)

            for point in left_eye:
                cv2.circle(image, tuple(point), 2, (0, 255, 0), -1)

            for point in right_eye:
                cv2.circle(image, tuple(point), 2, (0, 255, 0), -1)

            for point in mouth:
                cv2.circle(image, tuple(point), 2, (255, 0, 0), -1)

            eye_x1 = min(left_eye[:, 0])
            eye_y1 = min(left_eye[:, 1])
            eye_x2 = max(right_eye[:, 0])
            eye_y2 = max(right_eye[:, 1])

            eye_crop = image[
                max(0, eye_y1 - 20):min(h, eye_y2 + 20),
                max(0, eye_x1 - 20):min(w, eye_x2 + 20)
            ]

            mouth_x1 = min(mouth[:, 0])
            mouth_y1 = min(mouth[:, 1])
            mouth_x2 = max(mouth[:, 0])
            mouth_y2 = max(mouth[:, 1])

            mouth_crop = image[
                max(0, mouth_y1 - 20):min(h, mouth_y2 + 20),
                max(0, mouth_x1 - 20):min(w, mouth_x2 + 20)
            ]

            eye_pred = 0.0
            mouth_pred = 0.0

            if eye_crop.size != 0:

                eye_img = cv2.resize(eye_crop, (224, 224))
                eye_img = eye_img.astype("float32") / 255.0
                eye_img = np.expand_dims(eye_img, axis=0)

                eye_pred = eye_model.predict(
                    eye_img,
                    verbose=0
                )[0][0]

            if mouth_crop.size != 0:

                mouth_img = cv2.resize(mouth_crop, (224, 224))
                mouth_img = mouth_img.astype("float32") / 255.0
                mouth_img = np.expand_dims(mouth_img, axis=0)

                mouth_pred = mouth_model.predict(
                    mouth_img,
                    verbose=0
                )[0][0]

            score = (
                0.4 * (1 - ear) +
                0.3 * mar +
                0.15 * eye_pred +
                0.15 * mouth_pred
            )

            if ear < 0.23 and mar > 0.65:

                prediction = "DROWSY 😴"
                color = (0, 0, 255)

            elif ear < 0.23:

                prediction = "DROWSY 😴 (Eyes Closed)"
                color = (0, 0, 255)

            elif mar > 0.75:

                prediction = "DROWSY 😴 (Yawning)"
                color = (0, 0, 255)

            elif score > 0.55:

                prediction = "DROWSY 😴"
                color = (0, 0, 255)

            else:

                prediction = "AWAKE 👀"
                color = (0, 255, 0)

            cv2.putText(
                image,
                f"EAR: {ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"MAR: {mar:.2f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

            cv2.putText(
                image,
                prediction,
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3
            )

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Prediction Result",
                use_container_width=True
            )

            st.subheader("Prediction Details")

            st.write(f"EAR : {ear:.2f}")
            st.write(f"MAR : {mar:.2f}")
            st.write(f"Eye CNN Prediction : {eye_pred:.4f}")
            st.write(f"Mouth CNN Prediction : {mouth_pred:.4f}")
            st.write(f"Fusion Score : {score:.4f}")

            if "DROWSY" in prediction:
                st.error(prediction)
            else:
                st.success(prediction)

    else:

        st.error("❌ No Face Detected")
