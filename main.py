from trackers import HardhatTracker, PersonTracker
from utils import save_picture, read_picture, read_video, save_video


def main():
    # Picture detections
    # Model Setup
    hardhat_tracker = HardhatTracker(model_path='models/last_hardhat_200_epochs.pt')
    person_tracker = PersonTracker('yolov8x')

    # Input Reading
    input_picture_path = 'input_files/hardhat_input_picture_1.jpg'
    output_picture_path = 'output_files/hardhat_output_picture_1.jpg'
    picture = read_picture(input_picture_path)
    
    # Object Detection
    hardhat_detection = hardhat_tracker.detect_frame(input_picture_path)
    person_detection = person_tracker.detect_frame(input_picture_path)

    # Draw Boxes
    output_picture = hardhat_tracker.draw_frame_bboxes(picture, hardhat_detection)
    output_picture = person_tracker.draw_frame_bboxes(output_picture, person_detection, hardhat_detection)

    # Saving result
    save_picture(output_picture, output_picture_path)

    # Input Reading
    input_picture_path = 'input_files/hardhat_input_picture_2.jpg'
    output_picture_path = 'output_files/hardhat_output_picture_2.jpg'
    picture = read_picture(input_picture_path)

    # Object Detection
    hardhat_detection = hardhat_tracker.detect_frame(input_picture_path)
    person_detection = person_tracker.detect_frame(input_picture_path)

    # Draw Boxes
    output_picture = hardhat_tracker.draw_frame_bboxes(picture, hardhat_detection)
    output_picture = person_tracker.draw_frame_bboxes(output_picture, person_detection, hardhat_detection)

    # Saving result
    save_picture(output_picture, output_picture_path)

    # Video detections
    # Input Reading
    input_video_path = 'input_files/hardhat_input_video.avi'
    output_video_path = 'output_files/hardhat_output_video.avi'
    video = read_video(input_video_path)

    # Object Detection
    hardhat_detections = hardhat_tracker.detect_frames(video)
    hardhat_detections = hardhat_tracker.interpolate_hardhat_positions(hardhat_detections)
    person_detections = person_tracker.detect_frames(video)

    # Draw Boxes
    output_video_frames = hardhat_tracker.draw_video_bboxes(video, hardhat_detections)
    output_video_frames = person_tracker.draw_video_bboxes(output_video_frames, person_detections, hardhat_detections)
    
    # Saving result
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()
