
import torch
# from safe import PROJECT_NAME, BUCKET_NAME
from configs import DEPTH_MODEL_PATH, LR_MODEL_PATH, MR_MODEL_PATH
# from core import World
from models.yolo.utils import load_model
from models.depth_anything.utils import get_depth_model
from utils.general import load_gdf, crawl_project, save_hyperparams
from utils.trackers import *

import time
import logging
import argparse

from utils.shared_resources import initialize_depth_model, initialize_vision_models, initialize_trackers

# import geopandas as gpd
import pandas as pd
import os

def setup_logging(group, target_path):
    # Define the log directory based on the provided target_path and group
    log_dir = f'{target_path}/logs'
    
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure the root logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        filename='app.log',
                        filemode='w')
    root_logger = logging.getLogger()
    
    # Remove all handlers associated with the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add a FileHandler to log to a file within the specified directory
    log_file_path = os.path.join(log_dir, f'{group}.log')
    file_handler = logging.FileHandler(log_file_path, 'w')
    root_logger.addHandler(file_handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some data.")
    parser.add_argument('file_path', type=str, help='Path to the data file')
    parser.add_argument('target_path', type=str, help='Path to the folder which will contain the results')
    parser.add_argument('consolidated_world', type=str, help='Path to the most recent consolidated world representation.')
    args = parser.parse_args()

    group = args.file_path.split('/')[-1].split('.')[0]
    setup_logging(group, args.target_path)

    # Saving HyperParams
    project_path = './'
    output_file = args.target_path + '/hyperparams.json'

    all_hyperparams = crawl_project(project_path)
    save_hyperparams(all_hyperparams, output_file)

    # Initialize Trackers
    tracker_log_file_path = f'{args.target_path}/logs/trackers_{group}.log'
    initialize_trackers(tracker_log_file_path)
    save_tracker_logs()

    # Loading Models
    start = time.time()
    lr_model = load_model(LR_MODEL_PATH, device=torch.device('cuda:0'), half=True)

    mr_model = load_model(MR_MODEL_PATH , device=torch.device('cuda:0'), half=True)
    initialize_vision_models(lr_model, mr_model)
    
    depth_anything = get_depth_model('zoedepth', DEPTH_MODEL_PATH)
    initialize_depth_model(depth_anything)
    logging.info('Initialized DL models')

    # Loading target Locations360
    df = pd.read_csv(args.file_path)
    points = df[['LATITUDE', 'LONGITUDE']].values.tolist()

    from core.world import World
    world = World()
    try:
        world.loadstate(args.consolidated_world)
        logging.info('Found a savestate, loaded World.')
    except Exception as e:
        logging.info('No savestate found, using empty world.')


    # Creating Walks
    walk_counter_var = 1
    for index, (lat, lon) in enumerate(points):
        logging.info(f'########### location = {index} - {group} ###########')
        world.add_Location360(lat=lat, lon=lon, download=False, walk=False)
        location = world.fetch_location(lat=lat, lon=lon)
        location.download_starters()
        
        try:
            status = location.generate_walks(world=world)
            if status:  
                walk_counter_var += 1
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupt detected, exiting loop.')
            break
        except Exception as e:
            logging.info('ERROR DURING WALK GENERATION, SKIPPING LOCATION. THIS IS PROBLEMATIC. Exception: %s', exc_info=True)
        
        if walk_counter_var % 25 == 0:
            walked_locations = 0
            for _location in world.locations:
                if _location.walked == True:
                    walked_locations += 1
            world.savestate(f'{args.target_path}/intermediary_savestates/{group}_{str(walked_locations)}.pkl')
            save_tracker_logs()

    walked_locations = 0
    for _location in world.locations:
        if _location.walked == True:
            walked_locations += 1
    world.savestate(f'{args.target_path}/{group}_{str(walked_locations)}.pkl')
    world.savestate(f'{args.target_path}/intermediary_savestates/{group}_{str(walked_locations)}.pkl')