import os
import pickle
import shutil
import geopandas as gpd
import pandas as pd
from shapely import Point

from core import FOV
from core.location360 import Location360
from . import BUFFER_DISTANCE

import logging
logger = logging.getLogger()

class World:
    def __init__(self, buffer_distance=BUFFER_DISTANCE):
        self.locations = []
        self.gdf = gpd.GeoDataFrame(columns = ['id', 'lat', 'lon', 'image', 'starter', 'source', 'distance', 'image_type', 'target', 'geometry'], geometry='geometry')
        self.buffered_geometries = gpd.GeoDataFrame(columns=['id', 'geometry'], geometry='geometry')
        self.buffer_distance = buffer_distance

    def calculate_distance(self, point1, point2):
        return point1.distance(point2)

    def add_objects(self, objects):
        """
        Add new objects from a list of objects to the GeoDataFrame and update the buffer for each object.
        """
        # Generate data for all objects
        # print('starter being saved to gdf = ', [obj.starter for i, obj in enumerate(objects)])
        data = [{
            'id': self.gdf.index.max() + 1 + i if not self.gdf.empty else i,
            'obj': obj,
            'obj_type': obj.obj_type,
            'lat': obj.estimated_position.y,
            'lon': obj.estimated_position.x,
            'image': obj.image,
            'starter': obj.starter,
            'source': obj.image.filename,
            'distance': float(obj.estimated_distance),
            'image_type': obj.image.image_type,
            'target': obj.target,
            'geometry': Point(obj.estimated_position.x, obj.estimated_position.y)
        } for i, obj in enumerate(objects)]

        if len(data) == 0:
            # print('tried updating with no objects')
            return
        
        # Create a smaller GeoDataFrame with all the new objects
        new_rows = gpd.GeoDataFrame(data, geometry='geometry')

        # Concatenate the new rows with the existing GeoDataFrame
        if self.gdf.empty:
            self.gdf = new_rows
        else:
            self.gdf = pd.concat([self.gdf, new_rows], ignore_index=True).drop_duplicates()

        # Update buffered geometries for the new objects
        new_buffers = new_rows[['id', 'geometry']].copy()
        new_buffers['geometry'] = new_rows['geometry'].geometry.buffer(self.buffer_distance)
        self.buffered_geometries = pd.concat([self.buffered_geometries, new_buffers], ignore_index=True)
        self.buffered_geometries = self.buffered_geometries[self.buffered_geometries['id'].isin(self.gdf['id'])]

    def nearby_objects(self, obj):
        """
        Return the subset of self.gdf where the buffered geometries are within 
        the distance threshold of the passed object.
        """
        if self.buffered_geometries.empty:
            return gpd.GeoDataFrame(columns = self.gdf.columns, geometry='geometry')  # Return an empty GeoDataFrame if no buffered geometries exist

        # Identifying indices where the intersection occurs. Somewhat janky code.
        intersecting_indices = self.buffered_geometries['geometry'].intersects(obj.estimated_position)#.index
        ids = self.buffered_geometries.loc[intersecting_indices, 'id'].values

        return self.gdf[self.gdf['id'].isin(ids)]
    
    def verify_location_existence(self, locations, lat, lon):
        """
        Verify if the locations are already in the world.
        """
        for location in locations:
            if location.lat == lat and location.lon == lon:
                return True
        return False

    def add_Location360(self, lat, lon, obj=None, download=False, walk=False):
        if self.verify_location_existence(self.locations, lat, lon):
            logger.info('Location already exists in the world, will not add it again.')
            return
        if not obj:
            location = Location360(lat, lon, FOV)
        else:
            location = obj
        if download:
            location.download_starters(force=False)
        if walk:
            location.generate_walks(self, world=self)
        self.locations.append(location)

    def fetch_location(self, lat, lon):
        for location in self.locations:
            if location.lat == lat and location.lon == lon:
                return location

    def save_world_final_views(self, path='./tmp_data/', mode='focus'):
        # Check if path exists, if so, delete the contents
        if os.path.exists(path):
            shutil.rmtree(path)
        # Recreate the directory
        os.makedirs(path, exist_ok=True)
        
        counter = 0
        for location in self.locations:
            for starter in location.starters:
                for walk, status in zip(starter.walks, starter.walk_status):
                    for image in walk.images: 
                        if image.image_type == mode and status == True:
                            image = image.fetch_image()
                            # Assuming 'image' is a PIL Image object and requires a file format in the filename
                            image_file_path = os.path.join(path, f'image_{counter}.png')  # Adjust format as needed
                            image.save(image_file_path)
                            counter += 1

    def savestate(self, file_path):
        dir_path = os.path.dirname(file_path)
        if dir_path:  # If the directory path is not empty
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(self.__getstate__(), file)

    def loadstate(self, file_path):
        with open(file_path, 'rb') as file:
            state = pickle.load(file)
            self.__setstate__(state)

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'unpicklable_attribute' in state:
            print(state)
            del state['unpicklable_attribute']  # Example: Remove an unpicklable attribute
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
