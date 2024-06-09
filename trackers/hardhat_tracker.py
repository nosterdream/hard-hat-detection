from ultralytics import YOLO
import cv2
import pandas as pd


class HardhatTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def interpolate_hardhat_positions(self, hardhat_positions):
        # Interpolates hard hats positions
        hardhat_positions = [x.get(1, []) for x in hardhat_positions]
        df_hardhat_positions = pd.DataFrame(hardhat_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_hardhat_positions = df_hardhat_positions.interpolate()
        df_hardhat_positions = df_hardhat_positions.bfill()
        hardhat_positions = [{1: x} for x in df_hardhat_positions.to_numpy().tolist()]
        return hardhat_positions

    def detect_frames(self, frames):
        # Detects hard hats in series of frames
        hardhat_detections = []
        for frame in frames:
            hardhat_dict = self.detect_frame(frame)
            hardhat_detections.append(hardhat_dict)
        return hardhat_detections

    def detect_frame(self, frame):
        # Detects hard hats in a frame
        results = self.model.predict(frame)[0]
        hardhat_dict = {}
        i = 1
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            hardhat_dict[i] = result
            i = i + 1
        return hardhat_dict
    
    def draw_video_bboxes(self, video_frames, hardhat_detections):
        # Draws bounding boxes in series of frames
        output_video_frames = []
        for frame, hardhat_dict in zip(video_frames, hardhat_detections):
            frame = self.draw_frame_bboxes(frame, hardhat_dict)
            output_video_frames.append(frame)
        return output_video_frames
    
    def draw_frame_bboxes(self, frame, hardhat_detections):
        # Draws bounding boxes in a frame
        for track_id, bbox in hardhat_detections.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(frame, f"Hardhat ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        return frame
