# Object Detection Inference Interface (ODII)

ODII is a Python project designed to provide a common interface for inferring object detection models. This project supports a range of popular object detection   
models including YOLOX, YOLOv3, YOLOv4, YOLOv6, and YOLOv7.

## ✨ Features

- 🚀 **Unified Interface**: Interact with multiple object detection models using a single, easy-to-use interface.
- 🧹 **Reduced Boilerplate**: Simplifies the setup process by handling the installation of multiple models with varying instructions.
- 📚 **Lower Learning Curve**: Minimizes the complexity of understanding and writing inference code, making it easier to work with different models.
- 🔄 **Extensibility**: Easily extend the interface to support additional object detection models.

## 📦 Supported Models

- YOLOX  : https://github.com/Megvii-BaseDetection/YOLOX
- YOLOv3 : https://github.com/eriklindernoren/PyTorch-YOLOv3
- YOLOv4 : https://github.com/Tianxiaomo/pytorch-YOLOv4
- YOLOv6 : https://github.com/meituan/YOLOv6
- YOLOv7 : https://github.com/WongKinYiu/yolov7

## 📦 Reference for COCO Pretrained Weights

- [YOLOX](src/odii/yolox/readme.md)
- [YOLOv3](src/odii/yolov3/readme.md)
- [YOLOv4](src/odii/yolov4/readme.md)
- [YOLOv6](src/odii/yolov6/readme.md)
- [YOLOv7](src/odii/yolov7/readme.md)

## 🛠️ Requirements

- Python >= 3.8
- pip >= 24.2

## 📥 Installation

1. **Install PyTorch**: Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install the appropriate version of PyTorch for your system.

   For example, using pip:

   ```bash
   pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
   ```

2. **Clone the Repository and Install Dependencies**:

   ```bash
   git clone https://github.com/yourusername/odii.git
   cd odii
   python -m pip install -e .
   ```

## 🛠️ Usage

Here is an example of how to use ODII to run inference on an image:

```python
from odii import INFERENCE, plot_results, load_classes, load_yaml

# Load the classnames
classnames = load_classes('coco.names') # ['person','bicycle','car', ... ]

# Set the model paths & configs
model_config = {'yolov7': {'weights': 'weights/yolov7/yolov7.pt',
                           'config': None},
                'yolov4': {'weights': 'weights/yolov4/yolov4.weights',
                           'config': 'weights/yolov4/yolov4.cfg'},}
# Set Device
device = 'cuda'

# Input image path
image_path = 'tests/images/test2.jpg'

# --- Infer yolov7 model ---

model_name = 'yolov7' 

INF = INFERENCE(model_name=model_name,
                device=device,
                model_paths={'weights': model_config[model_name]['weights'],
                              'config': model_config[model_name]['config']})

yolov7_result = INF.infer_image(image_path=image_path,
                         confidence_threshold=0.4,
                         nms_threshold=0.4)


# --- Infer yolov4 model ---

model_name = 'yolov4' 

INF = INFERENCE(model_name=model_name,
                device=device,
                model_paths={'weights': model_config[model_name]['weights'],
                              'config': model_config[model_name]['config']})

yoloxm_result = INF.infer_image(image_path=image_path,
                         confidence_threshold=0.4,
                         nms_threshold=0.4)
```
More details for inference can be found in this notebook : [inference_demo.ipynb](inference_demo.ipynb)

## 📊 Results Format
The inference results are returned as a dictionary with the following format:

```python
{
    'boxes': [
        [74, 11, 900, 613],
        [77, 149, 245, 361],
        [560, 359, 737, 565],
        [139, 38, 414, 610]
    ],
    'scores': [
        0.8257260322570801,
        0.8446129560470581,
        0.8616959452629089,
        0.9366706013679504
    ],
    'classes': [2, 16, 28, 0]
}
```

## 🙏 Acknowledgements

1. https://github.com/Megvii-BaseDetection/YOLOX
2. https://github.com/Tianxiaomo/pytorch-YOLOv4
3. https://github.com/meituan/YOLOv6
4. https://github.com/WongKinYiu/yolov7
5. https://github.com/eriklindernoren/PyTorch-YOLOv3

