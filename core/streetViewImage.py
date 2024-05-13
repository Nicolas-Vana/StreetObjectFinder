import os
import random
import string

import numpy as np
import torch
import torchvision.transforms as T

from IPython.display import display

from env_vars import BUCKET_NAME, BUCKET_PATH

from models.depth_anything.utils import infer_depth_anything
from models.yolo.utils import convert_model_output, yolo_inference

from core.onImageObjectOfInterest import OnImageObjectOfInterest
# from core.Walk import Walk
from core import LR_MODEL_THRESHOLD, MR_MODEL_THRESHOLD, DEVICE, NUM_CLASSES_MR, IOU_THRESHOLD, IOU_THRESHOLD_REMOVAL, MAXIMUM_TRUSTED_DISTANCE

from utils.depth_map import plot_depth_map
from utils.object_identification import visualize_predictions
from utils.trackers import time_tracker
# from utils.shared_resources import lr_model, mr_model, depth_model
from utils.street_view import download_image_if_exists, parse_image_filename

import utils.shared_resources as sr

import logging
logger = logging.getLogger()

# Access models like this
lr_model = sr.lr_model
mr_model = sr.mr_model
depth_model = sr.depth_model

class StreetViewImage:
    def __init__(self, filename, metadata, image_type, walk=None):
        # Required Info
        self.is_available = False # Kinda deprecated
        self.filename = filename
        self.metadata = metadata
        self.image_type = image_type
        # if isinstance(walk, Walk):
        if type(walk).__name__ == 'Walk':
            self.starter = walk.images[0]

        # API Call Information
        lat_poste, lon_poste, fov, pitch, heading = parse_image_filename(filename)
        self.lat_call = lat_poste
        self.lon_call = lon_poste
        self.fov = fov
        self.pitch = pitch
        self.heading = heading
        self.size = 640

        # Metadata Information
        self.lat_image = metadata['location']['lat']
        self.lon_image = metadata['location']['lng']

        # Objects that will be added
        self.prediction = []
        self.mr_prediction = []
        self.objects = []
        

    def predict(self):
        if len(self.objects) != 0:
            logger.info('trying to predict for an image that has already been predicted')

        image = self.fetch_image()
        # TODO refactor this var name? call it lr_predictions and mr_predictions?
        self.prediction = yolo_inference(lr_model, device=DEVICE, image=image)
        self.prediction = convert_model_output(self.prediction)
        self.prediction = self.filter_predictions(self.prediction, LR_MODEL_THRESHOLD)
        self.prediction = self.filter_overlapping_predictions_boxes(self.prediction)

        self.mr_prediction = yolo_inference(mr_model, device=DEVICE, image=image)
        self.mr_prediction = convert_model_output(self.mr_prediction)
        self.mr_prediction = self.filter_predictions(self.mr_prediction, MR_MODEL_THRESHOLD)
        self.areas_of_interest, self.mr_prediction = self.split_mr_predictions_by_labels(self.mr_prediction)
        self.mr_prediction = self.filter_overlapping_predictions_boxes(self.mr_prediction)
        
        # Given the computational complexity of the depth map, we only compute it if there are objects any objects detected with score greater than threshold. 
        if len(self.prediction['scores']) > 0 and np.max(self.prediction['scores'].numpy()) > LR_MODEL_THRESHOLD:
            depth_map = self.compute_depth_map(image)
            # logger.info(np.max(depth_map))

        for index, bbox in enumerate(self.prediction['boxes']):
            if self.prediction['scores'][index].item() < LR_MODEL_THRESHOLD:
                continue
            
            bbox = bbox.numpy()
            bbox_mr, obj_type, type_score = self.find_corresponding_mr_box(self.mr_prediction, bbox, mode='object')
            bbox_area_of_interest, _, _ = self.find_corresponding_mr_box(self.areas_of_interest, bbox, mode='area_of_interest')
            
            self.objects.append(OnImageObjectOfInterest(obj_type=obj_type, bbox=bbox, score=self.prediction['scores'][index].item(), type_score=type_score, image=self, starter=self.starter, rgb_image=image, depth_map=depth_map, bbox_area_of_interest=bbox_area_of_interest, mode='mixed'))


    def split_mr_predictions_by_labels(self, original_dict, group1_labels=[0], group2_labels=list(np.arange(1, NUM_CLASSES_MR))):
        group1_dict = {key: [] for key in original_dict.keys()}
        group2_dict = {key: [] for key in original_dict.keys()}

        boxes = original_dict['boxes']
        labels = original_dict['labels']
        scores = original_dict['scores']

        for i in range(len(labels)):
            label = labels[i].item()
            box = boxes[i].tolist()
            score = scores[i].item()

            if label in group1_labels:
                group1_dict['boxes'].append(box)
                group1_dict['labels'].append(label)
                group1_dict['scores'].append(score)
            elif label in group2_labels:
                group2_dict['boxes'].append(box)
                group2_dict['labels'].append(label)
                group2_dict['scores'].append(score)

        group1_dict['boxes'] = torch.tensor(group1_dict['boxes'])
        group1_dict['labels'] = torch.tensor(group1_dict['labels'])
        group1_dict['scores'] = torch.tensor(group1_dict['scores'])

        group2_dict['boxes'] = torch.tensor(group2_dict['boxes'])
        group2_dict['labels'] = torch.tensor(group2_dict['labels'])
        group2_dict['scores'] = torch.tensor(group2_dict['scores'])

        return group1_dict, group2_dict


    def filter_predictions(self, prediction, threshold):
        filtered_boxes = []
        filtered_labels = []
        filtered_scores = []

        for score, box, label in zip(prediction['scores'], prediction['boxes'], prediction['labels']):
            if score >= threshold:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_scores.append(score)
        
        if len(filtered_boxes) == 0:
            return {'boxes':torch.tensor([]), 'labels':torch.tensor([]), 'scores':torch.tensor([])}

        filtered_predictions = {
            'boxes': torch.stack(filtered_boxes),
            'labels': torch.tensor(filtered_labels),
            'scores': torch.tensor(filtered_scores)
        }
        
        return filtered_predictions

    def filter_overlapping_predictions_boxes(self, predictions):
        if len(predictions['boxes']) <= 1:
            return predictions

        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']

        keep_indices = []
        non_overlapping_indices = []

        for i in range(len(boxes)):
            overlapped = False
            for j in range(len(boxes)):
                if i == j:
                    continue
                iou = self.compute_iou(boxes[i], boxes[j])
                if iou > IOU_THRESHOLD_REMOVAL:
                    overlapped = True
                    if scores[i] > scores[j]:
                        keep_indices.append(i)
                    else:
                        keep_indices.append(j)
            if not overlapped:
                non_overlapping_indices.append(i)

        if not keep_indices:
            keep_indices = non_overlapping_indices
        else:
            for idx in non_overlapping_indices:
                if idx not in keep_indices:
                    keep_indices.append(idx)

        keep_indices = list(set(keep_indices))

        filtered_boxes = torch.from_numpy(np.array([boxes[idx] for idx in keep_indices]))
        filtered_labels = torch.from_numpy(np.array([labels[idx] for idx in keep_indices]))
        filtered_scores = torch.from_numpy(np.array([scores[idx] for idx in keep_indices]))

        return {'boxes': filtered_boxes, 'labels': filtered_labels, 'scores': filtered_scores}

    def compute_iou(self, box1, box2):
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

    def find_corresponding_mr_box(self, prediction, bbox, mode='object'):
        if len(prediction) == 0:
            return [0,0,0,0], -1, 0

        ious = self.compute_bboxes_overlap(prediction, bbox)
        
        # Check if any IOU is above the threshold and find the index of the highest IOU
        max_iou = max(ious) if ious else 0
        threshold = IOU_THRESHOLD if mode == 'object' else 0
        
        if max_iou > threshold:
            # Find the index of the highest IOU that is greater than the threshold
            max_iou_index = ious.index(max_iou)
            
            # Retrieve the corresponding bounding box, label, and score
            max_iou_bbox = prediction['boxes'][max_iou_index].tolist()
            max_iou_label = prediction['labels'][max_iou_index].item()
            max_iou_score = prediction['scores'][max_iou_index].item()
            return max_iou_bbox, max_iou_label, max_iou_score
        else:
            return [0,0,0,0], -1, 0 

    def compute_bboxes_overlap(self, prediction, bbox):
        ious = []
        for box in prediction['boxes']:
            box_list = box.tolist()
            iou = self.compute_iou(bbox, box_list)
            ious.append(iou)
        return ious

    @time_tracker
    def compute_depth_map(self, image, debug=False):
        transform_to_tensor = T.ToTensor()
        image_tensor = transform_to_tensor(image)

        image_tensor = image_tensor.unsqueeze(0)  # Now the shape is (1, C, H, W)
        image_tensor = image_tensor.to(DEVICE)
        
        depth_map = infer_depth_anything(depth_model, image_tensor)
        depth_map = depth_map.cpu().numpy().squeeze()

        if debug:
            plot_depth_map(depth_map)

        return depth_map

    def fetch_image(self):
        image_exists = download_image_if_exists(BUCKET_NAME, BUCKET_PATH + self.filename)
        if image_exists:
            self.is_available = True
            return image_exists
        else:
            self.is_available = False
            logger.info('Something very weird is happening')
            return None

    def remove_mapped_objects(self, world):
        indexes_to_drop = []
        for index, obj in enumerate(self.objects):
            nearby_objects = world.nearby_objects(obj)
            final_targets = nearby_objects[(nearby_objects['target'] == True) & (nearby_objects['image_type'] == 'final_step') & (nearby_objects['starter'] != self.starter)]
            if (not final_targets.empty and obj.estimated_distance < MAXIMUM_TRUSTED_DISTANCE):
                indexes_to_drop.append(index)
        return [obj for idx, obj in enumerate(self.objects) if idx not in indexes_to_drop]

    def add_objects_on_image(self, world):
        if (not self.prediction) and (not self.prediction_types):
            logger.info('No objects on image or trying to add objects to world without identifying first')
            return
        objects_cleaned = self.remove_mapped_objects(world=world)
        world.add_objects(objects_cleaned)

    # TODO: Streamline this function - Ideally, this function is unified with the function ''
    def visualize_predictions_image(self, threshold=0.0, show=True):
        image = self.fetch_image()
        if not self.prediction:
            self.prediction = yolo_inference(lr_model, device=DEVICE, image=image)
            self.prediction = convert_model_output(self.prediction)
        image_all_predictions = visualize_predictions(predictions=self.prediction, image=image, threshold=threshold)
        if show:
            display(image_all_predictions)
        return image_all_predictions

    def visualize_predictions_on_image(self, prediction, threshold=0.0, show=True, save=False):
        image = self.fetch_image()
        image_all_predictions = visualize_predictions(predictions=prediction, image=image, threshold=threshold)
        if show:
            display(image_all_predictions)
        if save:
            random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            file_name = f"{random_name}.png"

            target_dir = './tmp_data/'
            os.makedirs(target_dir, exist_ok=True)

            image_path = os.path.join(target_dir, file_name)
            image_all_predictions.save(image_path)

            return image_all_predictions
        return image_all_predictions
