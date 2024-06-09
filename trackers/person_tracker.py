from ultralytics import YOLO
import cv2
import sys
from utils.bbox_utils import hardhat_is_on
sys.path.append('utils')


class PersonTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        # Detects people in series of frames
        person_detections = []
        for frame in frames:
            person_dict = self.detect_frame(frame)
            person_detections.append(person_dict)
        return person_detections

    def detect_frame(self, frame):
        # Detects people in a frames
        results = self.model.track(frame)[0]
        id_name_dict = results.names
        person_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                person_dict[track_id] = result
        return person_dict
    
    def draw_video_bboxes(self, video_frames, person_detections, hardhat_detections=None):
        # Draws bounding boxes in series of frames
        output_video_frames = []
        for frame, person_dict, hardhat_dict in zip(video_frames, person_detections, hardhat_detections):
            frame = self.draw_frame_bboxes(frame, person_dict, hardhat_dict)
            output_video_frames.append(frame)
        
        return output_video_frames

    def draw_frame_bboxes(self, frame, person_detections, hardhat_detections=None):
        # Draws bounding boxes in a frame
        color = (0, 0, 255)
        for track_id, bbox in person_detections.items():
            x1, y1, x2, y2 = bbox
            if hardhat_detections is not None:
                for id_hh, bbox_hh in hardhat_detections.items():
                    if hardhat_is_on(bbox, bbox_hh, frame):
                        color = (0, 255, 0)
                        break
            cv2.putText(frame, f"Person ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        return frame