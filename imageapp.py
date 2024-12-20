import streamlit as st
import cv2
import sqlite3
import numpy as np
import supervision as sv
from ultralytics import YOLO
from inference import get_model

# Load model
PLAYER_DETECTION_MODEL = get_model(
    model_id="football-players-detection-3zvbc/11", 
    api_key="MV9lmTMpz1bESUNpgrLl"
)

# Constants
BALL_ID = 0
PLAYER_ID = 2

# SQLite setup
def init_db():
    conn = sqlite3.connect("detections.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            object_class TEXT,
            x1 REAL, y1 REAL, x2 REAL, y2 REAL
        )
    """)
    conn.commit()
    return conn

conn = init_db()

# Streamlit app
st.title("IngameAi - Image Processing")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    # Load the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Process Image"):
        st.write("Processing the image, please wait...")

        # Perform inference
        result = PLAYER_DETECTION_MODEL.infer(img, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        # Separate detections
        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        player_detections = detections[detections.class_id == PLAYER_ID]

        # Annotate the image
        ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF']),
            thickness=2
        )
        annotated_image = ellipse_annotator.annotate(scene=img, detections=player_detections)

        triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )
        annotated_image = triangle_annotator.annotate(scene=annotated_image, detections=ball_detections)

        st.write("Image processed successfully!")
        st.write("Detected players:", len(player_detections))
        st.write("Detected balls:", len(ball_detections))

        st.write("Here is the data " , player_detections)

        # Display annotated image
        st.image(annotated_image, caption="Processed Image", use_container_width=True)


        # Save detections to SQLite
        image_name = uploaded_image.name
        with conn:
            for det in player_detections:
                conn.execute("""
                    INSERT INTO detections (image_name, object_class, x1, y1, x2, y2)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (image_name, "player", *det.xyxy))
            for det in ball_detections:
                conn.execute("""
                    INSERT INTO detections (image_name, object_class, x1, y1, x2, y2)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (image_name, "ball", *det.xyxy))

        st.success("Processing complete and data saved to database!")
