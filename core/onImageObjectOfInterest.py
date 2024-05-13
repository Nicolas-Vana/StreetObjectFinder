# from cmath import asin, cos, sin
from math import atan2, degrees, radians, asin, cos, sin
from scipy import stats
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from shapely import Point
from shapely.ops import nearest_points

import cv2

from core import FOV, FOV_FOCUS_RELAXATION
from .utils import get_bounded_linear_behavior
# import utils.shared_resources as sr

import logging
logger = logging.getLogger()

# street_gdf = sr.street_gdf
# sindex = sr.sindex 

class OnImageObjectOfInterest:
    def __init__(self, obj_type, bbox, score, type_score, image, starter, bbox_area_of_interest, mode='mixed', rgb_image=None, depth_map=None, background_mask=None, debug=False):
        # Required Info
        self.obj_type = obj_type
        self.bbox = bbox
        self.score = score
        self.type_score = type_score
        self.image = image
        self.starter = starter
        self.bbox_area_of_interest = bbox_area_of_interest
        self.debug = debug
        self.target = False

        # bbox info extraction
        self.bbox_width, self.bbox_height = self.compute_bbox_dimensions()
        if self.image.image_type == 'final_step' and bbox_area_of_interest != [0,0,0,0]:
            self.bbox_centroid_x, self.bbox_centroid_y = self.compute_bbox_centroids(self.bbox_area_of_interest)
        else:
            self.bbox_centroid_x, self.bbox_centroid_y = self.compute_bbox_centroids(self.bbox)
        self.heading = self.compute_bbox_heading(image)

        ## Distance Estimation
        self.estimated_distance_zoe = self.estimate_distance_zoe(depth_map,  rgb_image, background_mask)
        # Manually Parameterized
        self.correction = get_bounded_linear_behavior(self.estimated_distance_zoe, activation=0, saturation=40, upper_bound=0.6, lower_bound=0.9, mode='decreasing')
        self.estimated_distance_zoe = self.estimated_distance_zoe * self.correction
        self.estimated_distance = self.estimated_distance_zoe
        
        # Manually Parameterized
        self.safety = get_bounded_linear_behavior(self.estimated_distance, 10, 50, 0.8, 0.5, mode='decreasing')

        self.estimated_position = self.estimate_position(image)
        self.estimated_position_safe = self.estimate_position(image, safety=self.safety)

        # Other property estimation based on distance
        self.mode = self.get_bbox_mode()
        self.fov = self.compute_fov_to_focus(self.mode)
        self.pitch = self.compute_pitch(self.mode)
        # self.closest_road, self.closest_point_road = self.find_closest_road()

    ######## General Functions ########

    def get_bbox_mode(self):
        if self.image.image_type == 'final_step' and self.bbox_area_of_interest != [0,0,0,0]:
            return 'area_of_interest'
        elif self.image.image_type == 'final_step':
            return 'backup'
        else:
            return 'non_final'

    # For now, because of the way we have organized things, only pitch needs to be updated.
    def update_estimates_no_movement(self):
        self.mode = self.get_bbox_mode()
        self.pitch = self.compute_pitch(self.mode)
        self.fov = self.compute_fov_to_focus(self.mode)

    def compute_bbox_dimensions(self):
        bbox_width = self.bbox[2] - self.bbox[0]
        bbox_height = self.bbox[3] - self.bbox[1]
        return bbox_width, bbox_height

    def compute_bbox_centroids(self, bbox):
        x_centroid = np.mean([bbox[2], bbox[0]])
        y_centroid = np.mean([bbox[3], bbox[1]])
        return x_centroid, y_centroid
    
    def compute_bbox_heading(self, image):
        angular_displacement = (self.bbox_centroid_x - (image.size / 2)) * (image.fov / image.size)
        new_heading = (image.heading + angular_displacement) % 360

        return int(np.round(new_heading, 0))

    def compute_pitch(self, mode='area_of_interest'):
        if mode == 'non_final':
            return 0

        elif mode == 'area_of_interest':
            x1, y1, x2, y2 = self.bbox_area_of_interest
            bbox_center_y = (y1 + y2) / 2

            angular_displacement = ((self.image.size / 2) - bbox_center_y) * (self.image.fov / self.image.size)
            new_pitch = (self.image.pitch + angular_displacement)

            return int(np.round(new_pitch, 0))
        
        # Manually Parameterized
        else:
            max_dist, min_dist, lower, upper, exponent = (30, 5, 5, 40, 0.25)
            
            if self.estimated_distance <= min_dist:
                return upper
            elif self.estimated_distance >= max_dist:
                return lower
            else:
                normalized_distance = (self.estimated_distance - min_dist) / (max_dist - min_dist)
                pitch = upper - (upper - lower) * (normalized_distance ** exponent)
                return pitch


    def compute_fov_to_focus(self, mode='area_of_interest'):
        if mode == 'non_final':
            return FOV

        elif mode == 'area_of_interest':
            # Whats done here is a linear apporximation of a non-linear behavior, however this approximation is good, always having error < 10%, which is much greater than the precision of mostly anything thats done here.
            x1, y1, x2, y2 = self.bbox_area_of_interest
            img_size = self.image.size
            
            max_side_of_interest = np.max([np.abs(x1 - x2), np.abs(y1 - y2)])
            rounded_area_of_interest = max_side_of_interest 

            fov = rounded_area_of_interest*90/(img_size)
            
            return FOV_FOCUS_RELAXATION * fov
        
        # Manually Parameterized
        else:
            max_dist, min_param, lower, higher = 40, 10, 10, 30
            if self.estimated_distance <= min_param:
                return higher
            elif self.estimated_distance >= max_dist:
                return lower
            else:
                return higher - (higher - lower) * ((self.estimated_distance - min_param) / (max_dist - min_param))

    def estimate_position(self, image, safety=1.0):
        R = 6371000.0
        lat_image = radians(image.lat_image)
        lon_image = radians(image.lon_image)
        heading = radians(self.heading)

        new_lat = asin(sin(lat_image) * cos(self.estimated_distance*safety / R) + cos(lat_image) * sin(self.estimated_distance*safety / R) * cos(heading))
        new_lon = lon_image + atan2(sin(heading) * sin(self.estimated_distance*safety / R) * cos(lat_image), cos(self.estimated_distance*safety / R) - sin(lat_image) * sin(new_lat))

        return Point(degrees(new_lon), degrees(new_lat))

    # def find_closest_road(self):
    #     # gdf_singleton = GeoDataFrameSingleton.get_instance()
    #     # roads_gdf = gdf_singleton.gdf
    #     # sindex = gdf_singleton.sindex


    #     possible_matches_index = list(sindex.nearest(self.estimated_position, 1))
    #     nearest_linestring = street_gdf.iloc[possible_matches_index[1]].geometry.iloc[0]
    #     closest_point = nearest_points(self.estimated_position, nearest_linestring)[1]
    #     return nearest_linestring, closest_point
    
    ######## Zoe Depth Estimation ########
    
    def show_image_with_mask(self, image, mask, cmap=None):
        mask_display = np.where(mask == 255, 1, 0).astype(np.uint8)
        masked_image = cv2.bitwise_and(image, image, mask=mask_display)

        if len(image.shape) == 3:
            if image.shape[2] == 3:
                masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

        if cmap:
            plt.imshow(masked_image, cmap=cmap, vmin=0, vmax=80)
        else:
            plt.imshow(masked_image)

        # plt.colorbar()
        plt.title("Image with Mask Applied")
        plt.axis('off')
        plt.show()

    def apply_edge_detection(self, image, threshold1, threshold2, use_bbox=True):
        image = np.array(image, dtype='uint8')
        x1, y1, x2, y2 = map(int, self.bbox)

        cmap = 'viridis'
        gray = image
            
        if use_bbox:
            x1, y1, x2, y2 = map(int, self.bbox)
            cropped_gray = gray[y1:y2, x1:x2]

            # Apply Histogram Equalization and Gaussian Sharpening
            cropped_gray = cv2.equalizeHist(cropped_gray)
            
            # Manually Parameterized
            blurred = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
            cropped_gray = cv2.addWeighted(cropped_gray, 2.5, blurred, -0.9, 0)

            edges = cv2.Canny(cropped_gray, threshold1, threshold2)
            mask_cropped = np.zeros_like(edges)

        # Find countours and create Mask
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(mask_cropped, contours, -1, (255), thickness=cv2.FILLED)

        full_mask = np.zeros_like(gray)
        full_mask[y1:y2, x1:x2] = mask_cropped

        image_with_contours = np.array(gray)
        cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0), 1, offset=(x1, y1))

        if self.debug:
            clear_mask = np.array(255*np.ones(image_with_contours[y1:y2, x1:x2].shape), dtype='uint8')
            self.show_image_with_mask(image_with_contours[y1:y2, x1:x2], clear_mask, cmap=cmap)

            clear_mask = np.array(255*np.ones(image_with_contours.shape), dtype='uint8')
            self.show_image_with_mask(image_with_contours, clear_mask, cmap=cmap)

        return full_mask

    def apply_morphological_ops(self, mask, kernel_closure, kernel_opening):
        kernel_closure = np.ones(kernel_closure, np.uint8)
        kernel_opening = np.ones(kernel_opening, np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_closure)

        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(mask, [cnt], 0, (255), thickness=cv2.FILLED)
    
        return mask

    # Manually Parameterized
    def kernel_size_based_on_bbox_dims(self, max_size=60, min_size=25, lower=7, higher=25):
        width = self.bbox_width
        if width <= min_size:
            return (lower, lower)
        if width >= max_size:
            return (higher, higher)
        else:
            size = int(lower + (higher - lower) * ((width - min_size) / (max_size - min_size)))
            return (size, size)
        
    def estimate_distance_zoe(self, depth_map,  rgb_image, background_mask, method='edge'):
        x1, y1, x2, y2 = map(int, self.bbox)
        if method=='edge':
            # Manually Parameterized
            mask_depth_edge = self.apply_edge_detection(depth_map, threshold1=250, threshold2=280)
            kernel_closure = self.kernel_size_based_on_bbox_dims()

            mask_depth_edge_clean = self.apply_morphological_ops(mask_depth_edge, kernel_closure = kernel_closure, kernel_opening = (1, 1))
            final_mask = mask_depth_edge_clean
            
        bin_width = 2
        mode = stats.mode(np.round(depth_map[final_mask == 255] / bin_width, 0) * bin_width)[0]
        if self.debug:
            self.show_image_with_mask(np.array(depth_map)[y1:y2, x1:x2], final_mask[y1:y2, x1:x2])
            sns.histplot(depth_map[final_mask == 255])
        if np.isnan(mode):
            if self.debug:
                # print('OVERMASKING! NO PIXELS KEPT! USING BBOX MODE')
                logger.info('OVERMASKING! NO PIXELS KEPT! USING BBOX MODE')
            mode = stats.mode(np.round(depth_map[y1:y2, x1:x2].flatten(), 0))[0]
            return int(mode)
        else:
            return int(mode)
        
        return estimated_distance