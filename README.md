![GitHub User's stars](https://img.shields.io/github/stars/nosterdream/hard-hat-detection)
![GitHub forks](https://img.shields.io/github/forks/nosterdream/hard-hat-detection)

# Hard hat Detection
![hardhat_video](https://github.com/nosterdream/hard-hat-detection/assets/134122257/9228d43d-d741-45c7-a68d-7ea2ae6e63dc)


## Overview

This project focuses on detecting hard hats on individuals in images and videos. Leveraging the power of the YOLOv8 model, the system is capable of identifying people and determining if they are wearing hard hats. The output is an annotated image or video where detected people and hard hats are highlighted with bounding boxes. If a person is wearing a hard hat, the bounding box around them will be green, otherwise, it will be red.

You can find models from the [link](https://drive.google.com/drive/folders/1E6vjbpqiOCytXphIZB0OnvG45-WvTkH7?usp=sharing). 
The model last_hardhat_200_epochs.pt have to be in the directory root\models and yolov8x.pt in the root directory.


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Image Input](#image-input)
  - [Video Input](#video-input)
- [Training](#training)
- [Model](#model)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Requirements

- Python 3.11
- Ultralytics 8.2.21
- OpenCV 4.9.0.80
- Torch 2.3.0+cu121
- Torchvision 0.18.0+cu121
- Roboflow 1.1.30
- Pandas
- Jupyter Notebook

## Installation

To run this project, you need to have Python 3.8+ installed. Follow these steps to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/nosterdream/hard-hat-detection.git
    cd hardhat-detection
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

First, you need to get an instance of `HardhatTracker` specifying the model path:
```Python
from trackers import HardhatTracker

hardhat_tracker = HardhatTracker(model_path='models/last_hardhat_200_epochs.pt')
```

### Image Input

To read the image, you have to specify `input_picture_path` and `output_picture_path` and use `read_picture` function:

```Python
from utils import read_picture

# Input Reading
input_picture_path = 'input_files/hardhat_input_picture_1.jpg'
output_picture_path = 'output_files/hardhat_output_picture_1.jpg'
picture = read_picture(input_picture_path)
```

To run the hard hat detection on the image, use the `detect_frame` and `draw_frame_bboxes` methods of `HardhatTracker`:

```Python
hardhat_detection = hardhat_tracker.detect_frame(input_picture_path) # Detects hard hats on the image
output_picture = hardhat_tracker.draw_frame_bboxes(picture, hardhat_detection) # Draws bounding boxes on the image
```

To save the output image, use the `save_picture` function:
```Python
from utils import save_picture

save_picture(output_picture, output_picture_path)
```

This will output the image with detected hard hats, saving it in the `output_files/` directory. For people detections use the same commands for `PersonTracker`.

### Video Input

To read the video, you have to specify `input_video_path` and `output_video_path` and use `read_video` function:

```Python
from utils import read_video

# Input Reading
input_video_path = 'input_files/hardhat_input_video.avi'
output_video_path = 'output_files/hardhat_output_video.avi'
video = read_video(input_video_path)
```

To run the hard hat detection on the video, use the `detect_frames` and `draw_video_bboxes` methods of `HardhatTracker`:

```Python
hardhat_detections = hardhat_tracker.detect_frames(video) # Detects hard hats on the video
output_video_frames = hardhat_tracker.draw_video_bboxes(video, hardhat_detections) # Draws bounding boxes on the video
```

You can also use interpolation for better detections of hard hats after `detect_frames` method:

```Python
hardhat_detections = hardhat_tracker.detect_frames(video)
hardhat_detections = hardhat_tracker.interpolate_hardhat_positions(hardhat_detections)
```

To save the output image, use the `save_video` function:
```Python
from utils import save_video

save_video(output_video_frames, output_video_path)
```

This will output the video with detected hard hats, saving it in the `output_files/` directory. For people detections use the same commands for `PersonTracker`.

## Training

If you wish to train the model on the other dataset from Roboflow, follow these steps:
1. Get the dataset download code from https://universe.roboflow.com/
2. Change the code cell in the training/datasets/download_and_train.ipynb and run it to get dataset:

```Python
# Download the dataset from Roboflow Example
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("street-hazard-wquul").project("helmet-tracking")
version = project.version(2)
dataset = version.download("yolov8")
```
3. Run the following cell code, specifying the path to .yaml file, the number of epochs and the image size:

```Python
# Train YOLOv8 model for 100 epochs
from ultralytics import YOLO

model = YOLO('yolov8x.pt')
results = model.train(data="/kaggle/working/helmet-tracking-2/data.yaml", epochs=100, imgsz=640)
```
4. You can find the result in runs/detect/train/weights/.

This will train the YOLOv8 model on your dataset.

## Model

The model used in this project is based on YOLOv8. The pretrained weights for detecting hard hats can be found in the `models/` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## References

Model was trained on hard hat dataset from Roboflow: https://universe.roboflow.com/street-hazard-wquul/helmet-tracking

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on GitHub or contact the project maintainer at novoselov.g.v@mail.ru.

---
