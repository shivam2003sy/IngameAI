import streamlit as st
import cv2
import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from inference import get_model

# Load model
PLAYER_DETECTION_MODEL = get_model(
    model_id="football-players-detection-3zvbc/11", 
    api_key="MV9lmTMpz1bESUNpgrLl"
)

# Constants
BALL_ID = 0
PLAYER_ID = 2

# Streamlit app
st.title("IngameAi")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    # Save uploaded file to a temporary directory
    input_video_path = "input_video.mp4"
    with open(input_video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.video(input_video_path, format="video/mp4", start_time=0)

    if st.button("Process Video"):
        st.write("Processing the video, please wait...")

        # Annotators
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
            base=25,
            height=21,
            outline_thickness=1
        )

        # Tracker
        tracker = sv.ByteTrack()
        tracker.reset()

        # Frame generator
        frame_generator = sv.get_video_frames_generator(input_video_path,
                                                        #  stride=0
                                                         )

        # Get video information
        video_info = sv.VideoInfo.from_video_path(input_video_path)
        st.write(video_info)
        output_video_path = "output_video.mp4"
        
        # Process video
        with sv.VideoSink(target_path=output_video_path, video_info=video_info) as sink:
            for frame in frame_generator:
                result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
                detections = sv.Detections.from_inference(result)

                ball_detections = detections[detections.class_id == BALL_ID]
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                all_detections = detections[detections.class_id != BALL_ID]
                all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
                all_detections = tracker.update_with_detections(detections=all_detections)

                labels = [
                    f"#{tracker_id}"
                    for tracker_id
                    in all_detections.tracker_id
                ]

                all_detections.class_id = all_detections.class_id.astype(int)

                # Annotate frame
                annotated_frame = frame.copy()
                annotated_frame = ellipse_annotator.annotate(
                    scene=annotated_frame,
                    detections=all_detections)
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=all_detections,
                    labels=labels)
                annotated_frame = triangle_annotator.annotate(
                    scene=annotated_frame,
                    detections=ball_detections)

                # Write frame
                sink.write_frame(annotated_frame)

        st.success("Processing complete! Download the processed video below.")
        st.video(output_video_path, format="video/mp4", start_time=0)

        # Download link for the output video
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
