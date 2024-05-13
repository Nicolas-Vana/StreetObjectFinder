from PIL import Image, ImageDraw
import numpy as np

def compute_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    - box1: (xmin, ymin, xmax, ymax) for the first box.
    - box2: (xmin, ymin, xmax, ymax) for the second box.
    
    Returns:
    - iou: The IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    
    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area
    
    # Compute the IoU
    iou = intersection_area / union_area
    return iou

def visualize_predictions(predictions, image_path=None, image=None, threshold=0.5):
    if image_path:
      image = Image.open(image_path)
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    if predictions and 'boxes' in predictions and 'scores' in predictions:
        boxes = predictions['boxes']
        scores = predictions['scores']
        for i, box in enumerate(boxes):
            if scores[i] >= threshold:
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
                draw.text((box[0], box[1]), f'Score: {scores[i]:.2f}', fill="red")
    return image_copy
