from env_vars import API_KEY, BUCKET_NAME, BUCKET_PATH
from configs import POPULATE_DATASETS
from core.StarterImage import StarterImage
from utils.general import clear_location_dataset_folder
from utils.street_view import create_filename_from_params, fetch_and_save_image_with_metadata

import logging
logger = logging.getLogger()

class Location360:
    def __init__(self, lat, lon, fov) -> None:
        if 360 % fov != 0:
            raise ValueError("Invalid fov value. Only divisors of 360 are allowed as a fov value.")
        
        self.lat = lat
        self.lon = lon
        self.fov = fov

        self.headings = [fov*mult + 1 for mult in range(int(360/fov))]
        self.starters = []

        self.walked = False

    def download_starters(self, force=False):
        if self.walked:
            logger.info('Have already walked for this location, skipping starter downloading.')
            return
        for heading in self.headings:
            filename = create_filename_from_params(self.lat, self.lon, self.fov, pitch=0, heading=heading)
            image, metadata = fetch_and_save_image_with_metadata(lat=self.lat, lon=self.lon, api_key=API_KEY, bucket_name=BUCKET_NAME, bucket_path=BUCKET_PATH, filename=filename, fov=self.fov, pitch=0, heading=heading, force=force)
            try:            
                self.starters.append(StarterImage(filename, metadata, 'starter', self))
            except Exception as e:  # Catch all other exceptions not previously caught
                logger.info("metadata for this image = %s. The params for the call were: lat = %s, lon = %s.", str(metadata), str(self.lat), str(self.lon))
                logger.info("Unexpected error when downloading starter: %s. Skipping Starter.", str(e))
                continue
            
    def generate_walks(self, world):
        # status = False
        if self.walked:
            logger.info('Have already walked for this location, skipping walk generation.')
            return False
        if POPULATE_DATASETS:
            clear_location_dataset_folder(self.lat, self.lon)

        if len(self.starters) != 4:
            logger.info('Trying to generate walks without downloading the starters first. Or something much worse (not 0 and not 4)!')
            return False
        for index, starter in enumerate(self.starters):
            logger.info(f'Generating Walks for Starter %s', str(index))
            starter.generate_walks(world=world)
        
        self.walked = True
        return True
        
        

