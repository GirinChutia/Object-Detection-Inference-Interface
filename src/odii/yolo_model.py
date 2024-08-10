import os
import sys
import torch
import numpy as np
from typing import Tuple
from typing import Any
import cv2
import yaml
from .plotting_utils import plot_results

src_path = os.path.dirname(os.path.abspath(__file__))

# env : yolov5_env
yolox_exp_paths = {
    "yolox_s": f"{src_path}/yolox/exps/default/yolox_s.py",
    "yolox_m": f"{src_path}/yolox/exps/default/yolox_m.py",
    "yolox_l": f"{src_path}/yolox/exps/default/yolox_l.py",
    "yolox_x": f"{src_path}/yolox/exps/default/yolox_x.py",
}

yolox_models = ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]


def sanity_check_model_name(model_name):
    if model_name not in [
        "yolox_s",
        "yolox_m",
        "yolox_l",
        "yolox_x",
        "yolov3",
        "yolov7",
        "yolov4",
        "yolov6",
    ]:
        raise ValueError(
            "model_name should be one of yolox, yolov3, yolov7, yolov4, yolov6"
        )


def sanity_check_size_paths(model_paths, model_name):
    if model_name in ["yolox_s", "yolox_m", "yolox_l", "yolox_x"]:
        if model_paths["weights"] is None:
            raise ValueError("weights (.pth) path should not be none")
    if (model_name == "yolov3") or (model_name == "yolov4"):
        if model_paths["weights"] is None:
            raise ValueError("weights (.weights) path should not be none")
        if model_paths["config"] is None:
            raise ValueError("config (.cfg) path should not be none")
    if model_name == "yolov7":
        if model_paths["weights"] is None:
            raise ValueError("weights (.pt) path should not be none")


def add_to_sys_path(path):
    """Adds the given path to the system path."""
    if path is None or not os.path.exists(path):
        return
    sys.path.append(os.path.abspath(path))

def load_yaml(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def model_switch(model):
    """ Model switch to deploy status """
    from yolov6.layers.common import RepVGGBlock

    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, torch.nn.Upsample) and not hasattr(
            layer, "recompute_scale_factor"
        ):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility


class YOLOv6Utils:
    def __init__(self):
        from yolov6.data.data_augment import letterbox

        self.letterbox = letterbox

    def process_image(self, img_src, img_size, stride, half):
        """Process image before image inference."""
        # img_src = bgr_image
        image = self.letterbox(img_src, img_size, stride=stride)[0]
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        """Rescale the output to the original image shape"""
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (
            (ori_shape[1] - target_shape[1] * ratio) / 2,
            (ori_shape[0] - target_shape[0] * ratio) / 2,
        )

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes


class LOAD_MODEL:
    """
    Load a YOLO model.

    Args:
        model_name (str, optional): The name of the model. Defaults to None.
        device (str, optional): The device to load the model on. Defaults to None.
        model_path (dict, optional): The path to the model weights and config. Defaults to { "weights": None, "config": None }.
    """

    def __init__(
        self,
        model_name=None,
        device=None,
        model_path={"weights": None, "config": None},
    ):
        self.device = device
        self.model_path = model_path
        self.model_name = model_name

    def load_model(self):
        if self.model_name in yolox_models:
            return self.load_yolox().eval()
        elif self.model_name == "yolov3":
            return self.load_yolov3().eval()
        elif self.model_name == "yolov7":
            return self.load_yolov7().eval()
        elif self.model_name == "yolov4":
            return self.load_yolov4().eval()
        elif self.model_name == "yolov6":
            return self.load_yolov6().eval()
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def load_yolox(self):
        add_to_sys_path(f"{src_path}/yolox")
        from yolox.exp import get_exp

        exp = get_exp(yolox_exp_paths[self.model_name])
        model = exp.get_model()
        ckpt_file = self.model_path["weights"]
        ckpt = torch.load(ckpt_file, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        model.to(self.device)
        print(f"{self.model_name} model loaded!")
        return model

    def load_yolov3(self):
        add_to_sys_path(f"{src_path}/yolov3")
        from pytorchyolo import models

        model = models.load_model(
            self.model_path["config"], self.model_path["weights"], device=self.device
        )
        print(f"{self.model_name} model loaded!")
        return model

    def load_yolov7(self):
        add_to_sys_path(f"{src_path}")
        add_to_sys_path(f"{src_path}/yolov7")
        from yolov7.models.experimental import attempt_load

        model = attempt_load(self.model_path["weights"], map_location=self.device)
        print(f"{self.model_name} model loaded!")
        return model

    def load_yolov4(self):
        # https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/demo.py
        add_to_sys_path(f"{src_path}")
        add_to_sys_path(f"{src_path}/yolov4")
        from tool.darknet2pytorch import Darknet

        model = Darknet(self.model_path["config"])
        model.load_weights(self.model_path["weights"])
        # model.print_network()
        if self.device == "cuda":
            model.cuda()
        print(f"{self.model_name} model loaded!")
        return model

    def load_yolov6(self):
        add_to_sys_path(src_path)
        from yolov6.layers.common import DetectBackend

        model = DetectBackend(self.model_path["weights"], device=self.device)
        model_switch(model.model)
        model.model.float()
        if self.device != "cpu":
            img_size = (640, 640)
            model(
                torch.zeros(1, 3, *img_size)
                .to(self.device)
                .type_as(next(model.model.parameters()))
            )
        print(f"{self.model_name} model loaded!")
        return model


class INFERENCE:
    """
    Class for inference.

    Args:
        model_name (str): Name of the model to load. Defaults to None.
        model_paths (dict): Dictionary containing paths to weights and config files. Defaults to {'weights': None, 'config': None}.
        device (str): Device to use for inference. Defaults to 'cpu'.

    """

    def __init__(
        self,
        model_name=None,
        model_paths={"weights": None, "config": None},
        device="cpu",
        classnames=None,
    ):
        sanity_check_model_name(model_name)
        sanity_check_size_paths(model_paths, model_name)

        self.model_name = model_name
        self.model_loaded = False
        self.device = device
        self.classnames = classnames
        self.model_loader = LOAD_MODEL(
            model_name=self.model_name, device=self.device, model_path=model_paths,
        )

    def infer_yolox(
        self,
        bgr_image,
        test_size=(640, 640),
        num_classes=80,
        conf_threshold=0.25,
        nms_threshold=0.45,
    ) -> dict:
        """
        Infer YOLOX model on a given image.

        Args:
            bgr_image (numpy.ndarray): The input image in BGR format.
            test_size (tuple, optional): The size of the input image for the model. Defaults to (640, 640).
            num_classes (int, optional): The number of classes in the dataset. Defaults to 80.
            conf_threshold (float, optional): The confidence threshold for object detection. Defaults to 0.25.
            nms_threshold (float, optional): The threshold for non-maximum suppression. Defaults to 0.45.

        Returns:
            dict: The result of the object detection, including the bounding boxes, class IDs, and scores.
        """
        if not self.model_loaded:
            self.model = self.model_loader.load_model()
            self.model_loaded = True

        from yolox.data.data_augment import ValTransform
        from yolox.utils import postprocess

        preprocess_transform = ValTransform(legacy=False)
        image_info = {"id": 0}
        height, width = bgr_image.shape[:2]
        image_info["height"] = height
        image_info["width"] = width
        image_info["raw_img"] = bgr_image
        resized_ratio = min(test_size[0] / height, test_size[1] / width)
        image_info["resized_ratio"] = resized_ratio
        preprocessed_image, _ = preprocess_transform(bgr_image, None, test_size)
        preprocessed_image = torch.from_numpy(preprocessed_image).unsqueeze(0).float()
        if self.device == "cuda":
            preprocessed_image = preprocessed_image.cuda()
        with torch.no_grad():
            model_outputs = self.model(preprocessed_image)
            model_outputs = postprocess(
                model_outputs,
                num_classes,
                conf_threshold,
                nms_threshold,
                class_agnostic=True,
            )
        model_output = model_outputs[0].cpu()
        bboxes = model_output[:, :4] / image_info["resized_ratio"]
        classes = model_output[:, 6]
        scores = model_output[:, 4] * model_output[:, 5]
        filtered_bboxes, filtered_scores, filtered_classes = [], [], []
        for i in range(len(bboxes)):
            if scores[i] < conf_threshold:
                continue
            filtered_bboxes.append(bboxes[i].tolist())
            filtered_scores.append(scores[i].item())
            filtered_classes.append(int(classes[i]))

        result = {
            "boxes": filtered_bboxes,  # xyxy
            "scores": filtered_scores,
            "classes": filtered_classes,
        }
        self._result = result
        return result

    def infer_yolov3(
        self, bgr_image, test_size=(416, 416), conf_threshold=0.25, nms_threshold=0.45,
    ) -> dict:
        """
        Perform object detection on the input image using YOLOv3 model.

        Args:
            bgr_image (numpy.ndarray): The input image in BGR format.
            test_size (tuple, optional): The size to which the image will be resized. Defaults to (416, 416).
            conf_threshold (float, optional): The confidence threshold for object detection. Defaults to 0.25.
            nms_threshold (float, optional): The non-maximum suppression threshold for object detection. Defaults to 0.45.

        Returns:
            dict: A dictionary containing the detected bounding boxes, scores, and classes. The keys are "boxes", "scores", and "classes".
        """
        if not self.model_loaded:
            self.model = self.model_loader.load_model()
            self.model_loaded = True

        from pytorchyolo import detect
        import torchvision.transforms as transforms

        img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        detections = detect.detect_image(
            self.model,
            img,
            img_size=test_size[0],
            conf_thres=conf_threshold,
            nms_thres=nms_threshold,
            device=self.device,
        )
        filtered_bboxes, filtered_scores, filtered_classes = [], [], []
        for x1, y1, x2, y2, conf, cls_pred in detections:
            if conf < conf_threshold:
                continue
            filtered_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            filtered_scores.append(float(conf))
            filtered_classes.append(int(cls_pred))
        result = {
            "boxes": filtered_bboxes,  # xyxy
            "scores": filtered_scores,
            "classes": filtered_classes,
        }
        self._result = result

        return result

    def infer_yolov7(
        self, bgr_image, test_size=(640, 640), conf_threshold=0.5, nms_threshold=0.2,
    ) -> dict:
        """
        Infer YOLOv7 model on a given image.

        Args:
            bgr_image (numpy.ndarray): The input image in BGR format.
            test_size (tuple, optional): The size to which the image will be resized. Defaults to (640, 640).
            conf_threshold (float, optional): The confidence threshold for object detection. Defaults to 0.5.
            nms_threshold (float, optional): The non-maximum suppression threshold for object detection. Defaults to 0.2.

        Returns:
            dict: A dictionary containing the detected bounding boxes, scores, and classes. The keys are "boxes", "scores", and "classes".
        """
        if not self.model_loaded:
            self.model = self.model_loader.load_model()
            self.model_loaded = True

        from yolov7.utils.datasets import letterbox
        from yolov7.utils.general import non_max_suppression, scale_coords

        img_size = test_size[0]
        stride = 32

        im0 = bgr_image.copy()
        bgr_image = letterbox(bgr_image, img_size, stride=stride)[0]
        bgr_image = bgr_image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        bgr_image = np.ascontiguousarray(bgr_image)
        bgr_image = torch.from_numpy(bgr_image).to(self.device)
        bgr_image = bgr_image.float()
        bgr_image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if bgr_image.ndimension() == 3:
            bgr_image = bgr_image.unsqueeze(0)
        with torch.no_grad():
            pred = self.model(bgr_image, augment=False)[0]
        pred = non_max_suppression(
            pred, conf_threshold, nms_threshold, classes=None, agnostic=True
        )
        filtered_bboxes, filtered_scores, filtered_classes = [], [], []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(
                    bgr_image.shape[2:], det[:, :4], im0.shape
                ).round()
                for pred in reversed(det):
                    pred = list(pred.cpu().numpy())
                    x1, y1, x2, y2, conf, cls_pred = (
                        pred[0],
                        pred[1],
                        pred[2],
                        pred[3],
                        pred[4],
                        pred[5],
                    )
                    if conf < conf_threshold:
                        continue
                    filtered_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
                    filtered_scores.append(float(conf))
                    filtered_classes.append(int(cls_pred))
        result = {
            "boxes": filtered_bboxes,  # xyxy
            "scores": filtered_scores,
            "classes": filtered_classes,
        }

        self._result = result
        return result

    def infer_yolov4(self, bgr_image, conf_threshold=0.5, nms_threshold=0.2) -> dict:
        """
        Infer using YOLOV4 model.

        Args:
            bgr_image (numpy.ndarray): Input image.
            conf_threshold (float, optional): Confidence threshold for detection.
            nms_threshold (float, optional): Non-maximum suppression threshold.

        Returns:
            dict: Detection results.
        """
        """Infer using YOLOV4 model."""
        if not self.model_loaded:
            self.model = self.model_loader.load_model()
            self.model_loaded = True

        from yolov4.tool.torch_utils import do_detect
        
        width = bgr_image.shape[1]
        height = bgr_image.shape[0]
        bgr_image = cv2.resize(bgr_image, (self.model.width, self.model.height))
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.device == "cuda":
            use_cuda = True
        else:
            use_cuda = False
        predictions = do_detect(
            self.model, bgr_image, conf_threshold, nms_threshold, use_cuda
        )[0]
        filtered_bboxes, filtered_scores, filtered_classes = [], [], []
        for i in range(len(predictions)):
            box = predictions[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            cls_conf = float(box[5])
            cls_id = int(box[6])
            if cls_conf > conf_threshold:
                filtered_bboxes.append([x1, y1, x2, y2])
                filtered_scores.append(cls_conf)
                filtered_classes.append(cls_id)

        result = {
            "boxes": filtered_bboxes,  # xyxy
            "scores": filtered_scores,
            "classes": filtered_classes,
        }
        self._result = result
        return result

    def infer_yolov6(
        self,
        bgr_image,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.2,
        img_size: int = 640,
    ) -> dict:
        """
        Infer YOLOv6 model on a given image.

        Args:
            bgr_image (numpy.ndarray): The input image in BGR format.
            conf_threshold (float, optional): The confidence threshold for object detection. Defaults to 0.5.
            nms_threshold (float, optional): The threshold for non-maximum suppression. Defaults to 0.2.
            img_size (int, optional): The size of the input image for the model. Defaults to 640.

        Returns:
            dict: The result of the object detection, including the bounding boxes, class IDs, and scores.
        """
        if not self.model_loaded:
            self.model = self.model_loader.load_model()
            self.model_loaded = True

        YOLO6Utils = YOLOv6Utils()
        from yolov6.utils.nms import non_max_suppression

        stride = self.model.stride
        img_src = bgr_image
        device = self.device
        img, img_src = YOLO6Utils.process_image(img_src, img_size, stride, half=False)
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]
        pred_results = self.model(img)
        det = non_max_suppression(
            pred_results, conf_thres=conf_threshold, iou_thres=nms_threshold
        )[0]
        det[:, :4] = YOLO6Utils.rescale(
            img.shape[2:], det[:, :4], img_src.shape
        ).round()
        det = det.detach().cpu().numpy()
        boxes = []
        classes = []
        scores = []
        for x1, y1, x2, y2, conf, cls in det:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            classes.append(int(cls))
            scores.append(float(conf))
        result = {"boxes": boxes, "scores": scores, "classes": classes}  # xyxy
        self._result = result
        return result

    def infer_image(
        self, image_path, confidence_threshold=0.5, nms_threshold=0.45
    ) -> dict:
        """
        Infers the specified model on an image.

        Args:
            image_path (str): Path to the image file.
            confidence_threshold (float, optional): Confidence threshold for detection.
            nms_threshold (float, optional): Non-maximum suppression threshold.

        Returns:
            dict: Detection results with keys 'boxes', 'scores', and 'classes'.
        """
        image = cv2.imread(image_path)
        self._image_path = image_path

        if self.model_name in yolox_models:
            output = self.infer_yolox(
                image, conf_threshold=confidence_threshold, nms_threshold=nms_threshold
            )
        elif self.model_name == "yolov3":
            output = self.infer_yolov3(
                image, conf_threshold=confidence_threshold, nms_threshold=nms_threshold
            )
        elif self.model_name == "yolov7":
            output = self.infer_yolov7(
                image, conf_threshold=confidence_threshold, nms_threshold=nms_threshold
            )
        elif self.model_name == "yolov4":
            output = self.infer_yolov4(
                image, conf_threshold=confidence_threshold, nms_threshold=nms_threshold
            )
        elif self.model_name == "yolov6":
            output = self.infer_yolov6(
                image, conf_threshold=confidence_threshold, nms_threshold=nms_threshold
            )
            
        self._result = output

        return output
    
    def visualize_predictions(
        self,
        save=False,
        save_path="output.jpg",
    ) -> None:
        """
        Visualizes the predictions on the image.
        """
        plot_results(self._result,
                     self._image_path, 
                     self.class_names,
                     save=save,
                     save_path=save_path)
