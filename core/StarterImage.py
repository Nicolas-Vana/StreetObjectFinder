# Subclass
from configs import POPULATE_DATASETS
from core import LR_MODEL_THRESHOLD, LARGEST_DISTANCE_TO_CHASE, MAXIMUM_TRUSTED_DISTANCE
from core.streetViewImage import StreetViewImage
from core.walk import Walk

import logging
logger = logging.getLogger()

class StarterImage(StreetViewImage):
    def __init__(self, filename, metadata, image_type, location):
        super().__init__(filename, metadata, image_type)
        self.walks = []
        self.walk_status = []
        self.location = location
        self.starter = self

    def generate_walks(self, world):
        if len(self.walks) != 0:
            logger.info('Trying to create a walk for a starter that has already walked. Will skip this walk. This may create an edge case problem where some of the walks here may not have completed in a prior run.')
            return
        
        # Identify Objects of Interest
        self.predict()
        for index, interest_object in enumerate(self.objects):
            nearby_objects = world.nearby_objects(interest_object)
            final_targets = nearby_objects[(nearby_objects['target'] == True) & (nearby_objects['image_type'] == 'final_step') & (nearby_objects['starter'] != self.starter)]

            # If the confidence is above the threshold AND if the distance is smaller than the greater distance to chase AND (the object is not already in the World or it is far enough away that our estimated position will be innacurate)
            if interest_object.score >= LR_MODEL_THRESHOLD and interest_object.estimated_distance_zoe <= LARGEST_DISTANCE_TO_CHASE and (final_targets.empty or interest_object.estimated_distance > MAXIMUM_TRUSTED_DISTANCE):
                logger.info('Creating a walk for an object')
                walk = Walk(starting_image=self.starter , target_index=index)
                run_result = walk.run(world)
                self.walks.append(walk)
                self.walk_status.append(run_result)

                if POPULATE_DATASETS and run_result:
                    self.walks[-1].save_walk_to_dataset()