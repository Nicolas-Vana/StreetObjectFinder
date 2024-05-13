from io import BytesIO
from itertools import zip_longest
from math import atan2, cos, degrees, radians, sin

import numpy as np

from env_vars import BUCKET_NAME, BUCKET_PATH, DATASET_NAME, API_KEY
from configs import  FETCH_FOCUS, FETCH_MACRO, DEBUG
from core.streetViewImage import StreetViewImage
from core.utils import get_bounded_linear_behavior
from utils.general import haversine
from utils.street_view import create_filename_from_params, fetch_and_save_image_with_metadata, request_and_parse_streetview_metadata, upload_image
from utils.trackers import call_counter

from . import *

import logging
logger = logging.getLogger()

class Walk:
    def __init__(self, starting_image, target_index) -> None:
        self.images = [starting_image]
        self.target_indexes = [target_index]
        
        self.threshold = LR_MODEL_THRESHOLD
        self.steps = MAXIMUM_WALK_STEPS

        # Starting Position
        self.estimated_position, self.estimated_position_safe, self.estimated_distance = self.update_current_position()
    
    @call_counter
    def save_walk_to_dataset(self):
        # Since not all images have a defined target (focus and macro do not) we need to first pad the target index arr
        self.target_indexes = [x if x is not None else None for x, y in zip_longest(self.target_indexes, self.images, fillvalue=None)]

        for target, sv_image in zip(self.target_indexes, self.images):
            metadata = sv_image.metadata
            filename = sv_image.filename
            image_type = sv_image.image_type
            image = sv_image.fetch_image()
            self.save_image_to_dataset(image=image, metadata=metadata, filename=filename, image_type=image_type)

            if target == None:
                continue
            x1, y1, x2, y2 = map(int, sv_image.objects[target].bbox)
            bbox = image.crop((x1, y1, x2, y2))
            self.save_image_to_dataset(image=bbox, metadata=metadata, filename=filename, image_type='bbox')
    
    def save_image_to_dataset(self, image, metadata, filename, image_type):
        datasets_path = '/'.join(BUCKET_PATH.split('/')[:-2]) + DATASET_NAME
        
        location_path = '/' + str(self.images[0].location.lat) + '_' + str(self.images[0].location.lon) + '/'
        walk_id = 'walk_' + str(len(self.images[0].walks)) + '/'
        starter_folder = self.images[0].filename[:self.images[0].filename.rfind('.')] + '/'
        if image_type == 'bbox':
            datasets_path += location_path + starter_folder + walk_id + 'bboxes/'
        else:
            datasets_path += location_path + starter_folder + walk_id
        filename_with_type = filename[:filename.rfind('.')] + '_' + image_type + '.png'

        image_bytes = BytesIO()
        try:
            image.save(image_bytes, format=image.format)
        except:
            image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        upload_image(image_bytes, BUCKET_NAME, datasets_path + filename_with_type, metadata=metadata)
            
    ######### Core Methods #########

    @call_counter
    def run(self, world):
        # Adding The Initial Objects to the World and Setting the Current Target
        self.images[-1].objects[self.target_indexes[-1]].target = True
        self.images[-1].add_objects_on_image(world=world)

        for i in range(self.steps):
            if i == self.steps - 1:
                step_successful = self.step(world=world, image_type='final_step')
            else:
                step_successful = self.step(world=world, image_type='step')
            if step_successful == 'final':
                # logger.info('no more movement planned, stopping walk and fetching final images')
                logger.info('no more movement planned, stopping walk and fetching final images')
                break
            if not step_successful:
                logger.info('Walk was not successful. Cutting it Short on step = %s', str(i))
                return False
            

        # It is tempting to change the order and group this conditionals but do not do this! The order of the operations matter! 
        if FETCH_FOCUS:
            image_focus, image_focus_pil, focus_metadata = self.focus()
        if FETCH_MACRO:
            image_macro, image_macro_pil, macro_metadata = self.macro()
        
        if FETCH_FOCUS:
            self.images.append(image_focus)
        if FETCH_MACRO:
            self.images.append(image_macro)        
        return True
    
    # Walk towards and Fetch New Image
    def step(self, world, image_type):
        next_image, image_pil, metadata = self.move(image_type) # IS NONEs IF NO MOVEMENT IS PLANNED
        
        # This is just a foolproof for things like network erros and whatnot. Not very relevant.
        if next_image == False:
            logger.info('API Call Failed for some reason. Returning False')
            return False

        if next_image:
            # Add image to walk
            self.images.append(next_image)

            # Predict image objects and stop if none are detected
            self.images[-1].predict()
            if len(self.images[-1].objects) == 0:
                logger.info('After a step, no objects were detected. Stopping the walk.')
                return False

            # Based on Predictions, Try to Determine which object on the Image is most Likely to be the target. Update current estimated position.
            self.determine_and_update_target_index()

            # If no object is above the threshold, stop the walk
            if self.target_indexes[-1] == None:
                return False
            # Else, update current position, update target object and add objects to world
            else:
                # Check for stopping walk on object already searched.
                nearby_objects = world.nearby_objects(self.images[-1].objects[self.target_indexes[-1]])
                final_targets = nearby_objects[(nearby_objects['target'] == True) & (nearby_objects['image_type'] == 'final_step') & (nearby_objects['starter'] != self.images[0])]
            
                # If the confidence is above the threshold, and (the object is not already in the World or it is far enough away that our estimated position will be innacurate)
                if (final_targets.empty or self.images[-1].objects[self.target_indexes[-1]].estimated_distance > MAXIMUM_TRUSTED_DISTANCE):
                    self.estimated_position, self.estimated_position_safe,  self.estimated_distance = self.update_current_position()
                    # logger.info(self.estimated_position.y, self.estimated_position.x)
                    
                    if self.compute_total_target_position_delta() > MAXIMUM_WALK_DISTANCE_DELTA:
                        logger.info('Object position estimate changed too much during walk. Stopping walk because we are probably following the wrong target.')
                        return False
                    if self.images[-1].objects[self.target_indexes[-1]].obj_type not in CLASSES_TO_CHASE and self.images[-1].objects[self.target_indexes[-1]].type_score > INTERRUPTION_CONFIDENCE:
                        logger.info('I believe the object that I am chasing is not of a relevant class. Stopping walk.')
                        # logger.debug('Detected Label for object that will no be chased =', self.images[-1].objects[self.target_indexes[-1]].obj_type)
                        print('Detected Label for object that will no be chased =', self.images[-1].objects[self.target_indexes[-1]].obj_type)
                        return False


                    self.images[-1].objects[self.target_indexes[-1]].target = True
                    self.images[-1].add_objects_on_image(world=world)
                    return True
                else:
                    logger.info('During the walk, I suspected that I have found this object already. stopping the Walk.')
                    return False

        else:
            # Reset the image type to reflect the fact that we are reusing the image. Add the objects with the new tags.
            logger.info('No movement planned, reusing image and setting image type to %s', str(image_type))
            self.images[-1].image_type = image_type
            self.images[-1].add_objects_on_image(world=world)
            self.images[-1].objects[self.target_indexes[-1]].update_estimates_no_movement()
            return 'final'

    def move(self, image_type):
        # Evaluate closest point to the estimated position (safe)
        metadata = request_and_parse_streetview_metadata(self.estimated_position_safe.y, self.estimated_position_safe.x, API_KEY)
        if metadata['status'] == 'ZERO_RESULTS':
            logger.info('No Metadata results for this metadata API call. This probably means there is no streetview image in a 50 meter range of the location.')
            return False, None, None
        next_lat, next_lon = np.round(metadata['location']['lat'], 6), np.round(metadata['location']['lng'], 6)

        # Check if no movement is planned
        if self.check_no_movement(next_lat, next_lon):
            return None, None, None
        
        heading = int(self.calculate_bearing(next_lat, next_lon, self.estimated_position.y, self.estimated_position.x))
        fov = int(self.images[-1].fov)
        pitch = int(self.images[-1].objects[self.target_indexes[-1]].pitch)
        
        # Create Image
        next_sv_image, image, metadata = self.create_StreetViewImage(next_lat, next_lon, fov, pitch, heading, image_type)
        if not isinstance(next_sv_image, StreetViewImage):
            return False, None, None
        
        return next_sv_image, image, metadata

    def focus(self):
        # There should be no difference in using either of the lines below, just keep whatever is currently working lmao
        # next_lat, next_lon = self.images[-1].lat_call, self.images[-1].lon_call
        next_lat, next_lon = self.images[-1].lat_image, self.images[-1].lon_image
        
        heading = int(self.images[-1].objects[self.target_indexes[-1]].heading)
        fov = int(self.images[-1].objects[self.target_indexes[-1]].fov)

        self.images[-1].objects[self.target_indexes[-1]].update_estimates_no_movement()
        pitch = int(self.images[-1].objects[self.target_indexes[-1]].pitch)
        
        next_sv_image, image, metadata = self.create_StreetViewImage(next_lat, next_lon, fov, pitch, heading, 'focus')
        if not isinstance(next_sv_image, StreetViewImage):
            return False, None, None
        
        return next_sv_image, image, metadata

    def macro(self):
        # There should be no difference in using either of the lines below, just keep whatever is currently working lmao
        # next_lat, next_lon = self.images[-1].lat_call, self.images[-1].lon_call
        next_lat, next_lon = self.images[-1].lat_image, self.images[-1].lon_image
        
        heading = int(self.images[-1].objects[self.target_indexes[-1]].heading)
    
        # TODO: This does not belong here, move to the OnImageObject class when possible 
        def compute_fov_macro(estimated_distance, max_dist=30, min_dist=2.5, lower=40, higher=120):
            if estimated_distance <= min_dist:
                return higher
            elif estimated_distance >= max_dist:
                return lower
            else:
                exponent = 0.5
                normalized_distance = (estimated_distance - min_dist) / (max_dist - min_dist)
                pitch = higher - (higher - lower) * (normalized_distance ** exponent)
                return pitch
        
        fov = compute_fov_macro(self.images[-1].objects[self.target_indexes[-1]].estimated_distance)
        # if DEBUG:
        #     logger.info('macro obj estimated distance =', self.images[-1].objects[self.target_indexes[-1]].estimated_distance, ', resulting fov =', fov)

        self.images[-1].objects[self.target_indexes[-1]].update_estimates_no_movement()
        pitch = MACRO_PITCH_ADJUSTMENT*int(self.images[-1].objects[self.target_indexes[-1]].pitch)

        next_sv_image, image, metadata = self.create_StreetViewImage(next_lat, next_lon, fov, pitch, heading, 'macro')
        if not isinstance(next_sv_image, StreetViewImage):
            return False, None, None
        
        return next_sv_image, image, metadata

    ######### Support Methods #########

    def determine_and_update_target_index(self):
        # Compute New Metrics To Assess likelihood of being the target
        self.distances, self.closest_object_index = self.compute_distances()
        self.angles, self.closest_angle_object_index = self.compute_centralizations()
        self.scores = [obj.score for obj in self.images[-1].objects]
        self.likelihoods = self.compute_likelihoods()

        # Choose most likely object
        self.target_indexes.append(self.update_target_index()) 
    
    # Distance Threshold Determines the Maximum Distance Delta From a Previous Step To The Current One
    def update_target_index(self):
        score_mask = np.array(self.scores) > self.threshold
        
        relative_maximum_step_distance_delta = get_bounded_linear_behavior(self.estimated_distance, 10, 30, 30, 10, mode='increasing')

        distance_mask = np.array(self.distances) < relative_maximum_step_distance_delta
        if sum(distance_mask) == 0:
            logger.info('Object position changed too much during step, cutting walk short.')
            target_index = None
        
        final_mask = (score_mask) & (distance_mask)
        filtered_likelihoods = np.array(self.likelihoods)[final_mask]

        if filtered_likelihoods.size > 0:
            max_likelihood_index = np.argmax(filtered_likelihoods)
            target_index = np.where(final_mask)[0][max_likelihood_index]
        else:
            logger.info('After a Step no object has score higher than the threshold and closer than the distance threshold. Stopping the walk.')
            target_index = None  # or any other default value or logic

        return target_index

    def compute_distances(self):
        smallest_distance = 100000
        distances = []
        for index, obj in enumerate(self.images[-1].objects):
            distance = haversine(self.estimated_position.y, self.estimated_position.x, obj.estimated_position.y, obj.estimated_position.x)
            distances.append(distance)
            if distance < smallest_distance:
                smallest_distance = distance
                closest_object_index = index
        return distances, closest_object_index
    
    def compute_centralizations(self, size=640):
        smallest_angle = 100000
        angles = []
        for index, obj in enumerate(self.images[-1].objects):
            angle = np.abs((size/2) - obj.bbox_centroid_x)
            angles.append(angle)
            if angle < smallest_angle:
                smallest_angle = angle
                closest_object_index = index
        
        return angles, closest_object_index
    
    def inverse_transformation(self, arr):
        """Compute scores using the inverse transformation method."""
        arr = np.array(arr)
        inverse_values = np.divide(1, arr, where=arr!=0)

        inverse_values[np.isinf(inverse_values)] = 0
        sum_of_inverse_values = np.sum(inverse_values)
        if sum_of_inverse_values != 0:
            normalized_scores = inverse_values / sum_of_inverse_values
        else:
            normalized_scores = np.zeros_like(inverse_values)
        return normalized_scores

    def compute_likelihoods(self):
        distance_likelihood = LIKELIHOOD_RATIO*self.inverse_transformation(self.distances)
        angle_likelihood = self.inverse_transformation(self.angles)
        
        return (distance_likelihood + angle_likelihood) / (LIKELIHOOD_RATIO + 1)

    def update_current_position(self):
        return self.images[-1].objects[self.target_indexes[-1]].estimated_position, self.images[-1].objects[self.target_indexes[-1]].estimated_position_safe, self.images[-1].objects[self.target_indexes[-1]].estimated_distance
    
    def compute_total_target_position_delta(self):
        first = self.images[0].objects[self.target_indexes[0]].estimated_position
        last = self.images[-1].objects[self.target_indexes[-1]].estimated_position
        
        distance = haversine(first.y, first.x, last.y, last.x)
        return distance

    def check_no_movement(self, lat, lon):
        if (lat == self.images[-1].lat_call) and (lon == self.images[-1].lon_call):
            return True
        else:
            return False

    def calculate_bearing(self, lat_source, lon_source, lat_target, lon_target):
        # Convert latitude and longitude from degrees to radians
        lat_source, lon_source, lat_target, lon_target = map(radians, [lat_source, lon_source, lat_target, lon_target])

        # Calculate the difference in longitude
        dlon = lon_target - lon_source

        # Calculate the bearing
        x = cos(lat_target) * sin(dlon)
        y = cos(lat_source) * sin(lat_target) - sin(lat_source) * cos(lat_target) * cos(dlon)
        bearing = atan2(x, y)

        # Convert bearing from radians to degrees and adjust to 0-360 degrees
        bearing = (degrees(bearing) + 360) % 360
        return int(np.round(bearing, 0))

    def create_StreetViewImage(self, lat, lon, fov, pitch, heading, image_type):
        # Quantization
        lat, lon = np.round(lat, 6), np.round(lon, 6)
        fov = int(round(fov / 3) * 3)
        pitch = int(round(pitch / 3) * 3)
        heading = int(round(heading / 3) * 3)

        filename = create_filename_from_params(lat, lon, fov, pitch, heading)
        image, metadata = fetch_and_save_image_with_metadata(lat=lat, lon=lon, api_key=API_KEY, bucket_name=BUCKET_NAME, bucket_path=BUCKET_PATH, filename=filename, fov=fov, pitch=pitch, heading=heading)
        
        if metadata == None:
            return None

        sv_image = StreetViewImage(filename, metadata, image_type, walk=self)
        return sv_image, image, metadata

    def traverse(self, threshold=None):
        for image in self.images:
            if threshold:
                image.visualize_predictions_image(threshold=threshold)
            else:
                image.visualize_predictions_image(threshold=self.threshold)
