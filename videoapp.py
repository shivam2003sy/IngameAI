import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import logging

from inference import get_model


# Load a pre-trained YOLOv8 model
model = get_model(model_id="football-player-detection-kucab-ofgzn/2", api_key="MV9lmTMpz1bESUNpgrLl")

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

# Function to initialize the YOLO models
def initialize_models( pitch_model_path):
    # player_model = YOLO(player_model_path).to(device='cuda')
    pitch_model = YOLO(pitch_model_path).to(device='cuda')
    return  pitch_model

# Function to process a video frame and apply detections
def process_frame(frame, pitch_model, player_model, tracker, box_annotator, label_annotator):
    # Pitch detection
    result_pitch = pitch_model(frame, verbose=False)[0]
    keypoints = sv.KeyPoints.from_ultralytics(result_pitch)

    # Player detections
    results = player_model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)

    # Annotate bounding boxes
    frame = box_annotator.annotate(scene=frame, detections=detections)

    # Create and annotate labels
    labels = [
        f"Tracker ID: {tracker_id} Class: {class_name}"
        for tracker_id, class_name in zip(detections.tracker_id, detections.data['class_name'])
    ]
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame, detections, keypoints

# Function to render radar updates
def render_radar_update(detections, keypoints, color_lookup):
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    for i in range(len(COLORS)):
        radar = draw_points_on_pitch(
            config=CONFIG,
            xy=transformed_xy[color_lookup == i],
            face_color=sv.Color.from_hex(COLORS[i]),
            radius=20,
            pitch=radar
        )
    return radar


import csv
# Main function to process video
def process_video(source_video_path, target_video_path_original, target_video_path_radar, csv_file_path, pitch_model_path, max_frames=1000):
    # Initialize models
    pitch_model = initialize_models(pitch_model_path)
    player_model = get_model(model_id="football-player-detection-kucab-ofgzn/2", api_key="MV9lmTMpz1bESUNpgrLl")

    # Video info and setup
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)


    count_frame = 0
    with sv.VideoSink(target_path=target_video_path_original, video_info=video_info) as sink , sv.VideoSink(target_path=target_video_path_radar, video_info=video_info) as sink_radar:
        for frame in frame_generator:
            count_frame += 1
            if count_frame > max_frames:
                break

            logging.info(f"Processing frame {count_frame}")

            # Process frame
            frame, detections, keypoints = process_frame(
                frame, pitch_model, player_model, tracker, box_annotator, label_annotator
            )

            # Radar visualization
            color_lookup = np.zeros(len(detections), dtype=int)
            radar = render_radar_update(detections, keypoints, color_lookup)

            # Overlay radar on frame
            # radar_h, radar_w, _ = radar.shape
            # frame_h, frame_w, _ = frame.shape
            # rect = sv.Rect(x=0, y=0, width=frame_w, height=frame_h)
            # annotated_frame = sv.draw_image(frame, radar, opacity=1, rect=rect)


            ellipse_annotator = sv.EllipseAnnotator(
                color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                thickness=2
            )
            label_annotator = sv.LabelAnnotator(
                color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                text_color=sv.Color.from_hex('#000000'),
                text_position=sv.Position.BOTTOM_CENTER
            )
            vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8)

            labels = [
                f"#{tracker_id}"
                for tracker_id
                in detections.tracker_id
            ]

            detections.class_id = detections.class_id.astype(int)

            annotated_frame2 = frame.copy()
            annotated_frame2 = ellipse_annotator.annotate(
                                scene=annotated_frame2,
                                detections=detections,
                            )
            annotated_frame2 = label_annotator.annotate(
                                scene=annotated_frame2,
                                detections=detections,
                                labels=labels
                            )
            annotated_frame2 = vertex_annotator.annotate(
                                scene=annotated_frame2,
                                key_points=keypoints,
                            )

            sv.plot_image(annotated_frame2)


            # Display frame
            # cv2_img = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            # cv2_imshow(cv2_img)

            # Save the annotated frames
            # sink.write_frame(annotated_frame)

            # Save radar-only frames
            cv2_img = cv2.cvtColor(annotated_frame2, cv2.COLOR_RGB2BGR)
            cv2_imshow(cv2_img)
            # sink_radar.write_frame(annotated_frame2)


