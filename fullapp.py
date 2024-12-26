import streamlit as st
import sqlite3
import numpy as np
import json
import supervision as sv
from inference import get_model

from PIL import Image
from io import BytesIO




from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
)

from sports.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()


from sports.common.view import ViewTransformer



# Load model
PLAYER_DETECTION_MODEL = get_model(
    model_id="football-players-detection-3zvbc/11", 
    api_key="MV9lmTMpz1bESUNpgrLl"
)
FIELD_DETECTION_MODEL = get_model(
    model_id="football-field-detection-f07vi/14",
    api_key="MV9lmTMpz1bESUNpgrLl"
    )

# Constants
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20, height=17
)


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


# Anotation tools
ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=20, height=17
)

# Streamlit app
st.title("IngameAi - Image Processing")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img =  Image.open(BytesIO(file_bytes))
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Process Image"):
        st.write("Processing... Please wait.") 

    result = PLAYER_DETECTION_MODEL.infer(img, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    players_detections = detections[detections.class_id == PLAYER_ID]

    annotated_image = ellipse_annotator.annotate(
        scene=img,
        detections=players_detections
    )

    annotated_image  = triangle_annotator.annotate(
        scene=annotated_image,
        detections=ball_detections
    )
    st.write("Image processed successfully!")
    st.write("Detected players:", len(players_detections))
    st.write("Detected balls:", len(ball_detections))

    st.image(annotated_image, caption="Processed Image.", use_container_width=True)

        # detect pitch key points

    result = FIELD_DETECTION_MODEL.infer(img, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    # project ball, players and referies on pitch

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]
    transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

        # visualize video game-style radar view

    annotated_frame = draw_pitch(CONFIG)
    annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_ball_xy,
            face_color=sv.Color.WHITE,
            edge_color=sv.Color.BLACK,
            radius=10,
            pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 2],
            face_color=sv.Color.from_hex('00BFFF'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_frame)
    annotated_frame = draw_points_on_pitch(
            config=CONFIG,
            xy=pitch_players_xy[players_detections.class_id == 2],
            face_color=sv.Color.from_hex('FF1493'),
            edge_color=sv.Color.BLACK,
            radius=16,
            pitch=annotated_frame)

    st.image(annotated_frame, caption="Processed Image.", use_container_width=True)

    

