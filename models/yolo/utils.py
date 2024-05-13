import torch
import numpy as np
import math

# from yolov6.data.data_augment import letterbox
from typing import List, Optional

# from yolov6.utils.events import LOGGER, load_yaml
from models.yolo import *

from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
# from .yolov6.layers.common import DetectBackend

def check_img_size(img_size, s=32, floor=0):
  def make_divisible( x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
  """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
  if isinstance(img_size, int):  # integer i.e. img_size=640
      new_size = max(make_divisible(img_size, int(s)), floor)
  elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
      new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
  else:
      raise Exception(f"Unsupported type of img_size: {type(img_size)}")

  if new_size != img_size:
      print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
  return new_size if isinstance(img_size,list) else [new_size]*2

def process_image(img_src, img_size, stride, half):
  '''Process image before image inference.'''
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))  # HWC to CHW
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src

def convert_model_output(output_tensor):
    # Ensure tensor is on CPU for further operations
    output_tensor = output_tensor.cpu()

    if output_tensor.dim() == 1:
        # Make it two-dimensional ([1, 6])
        output_tensor = output_tensor.unsqueeze(0)

    # Separate boxes, scores, and labels
    boxes = output_tensor[:, :4]
    scores = output_tensor[:, 4]
    labels = output_tensor[:, 5].long()  # Convert labels to long in case they are in another format

    # Create the output dictionary
    output_dict = {
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }

    return output_dict

def yolo_inference(model, image, device):
    image = np.asarray(image)
    stride = model.stride
    img, img_src = process_image(image, img_size, stride, half)
    img = img.to(device)

    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    classes:Optional[List[int]] = None # the classes to keep
    det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

    # This might be required if images with shape other than 640x640 are used - Nicolas
    # gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
    return det

def load_model(checkpoint, device, half, img_size=[640, 640]):
    model = DetectBackend(f"{checkpoint}.pt", device=device)
    stride = model.stride
    # class_names = ['poste'] # load_yaml("./data/coco.yaml")['names']

    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False
        
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

    img_size = check_img_size(img_size, s=stride)
    return model