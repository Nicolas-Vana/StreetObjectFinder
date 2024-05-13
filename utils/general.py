from math import radians, cos, sin, asin, sqrt

from google.cloud import storage
from google.cloud import bigquery

import pandas as pd
import geopandas as gpd
from shapely import wkt

import os
import pickle
import ast
import json

import logging

from env_vars import BUCKET_NAME, BUCKET_PATH, DATASET_NAME, SERVICE_ACCOUNT_JSON_PATH
logger = logging.getLogger(__name__)

################ GDF LOADING ################

def init_clients(service_account_json_path):
  storage_client = storage.Client.from_service_account_json(service_account_json_path)
  bigquery_client = bigquery.Client.from_service_account_json(service_account_json_path)
  return storage_client, bigquery_client

storage_client, bigquery_client = init_clients(SERVICE_ACCOUNT_JSON_PATH)

def clear_location_dataset_folder(lat, lon):
    bucket = storage_client.bucket(BUCKET_NAME)
    
    datasets_path = '/'.join(BUCKET_PATH.split('/')[:-2]) + DATASET_NAME
    location_path = '/' + str(lat) + '_' + str(lon) + '/'
    folder_path = datasets_path + location_path

    blobs = bucket.list_blobs(prefix=folder_path)
    for blob in blobs:
        blob.delete()

################ OTHER ################

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000 # Radius of earth in meters
    return c * r

################ HYPERPARAMS ################

def extract_hyperparams(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    hyperparams = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                if isinstance(node.value, (ast.Num, ast.Str, ast.NameConstant)):
                    var_value = node.value.n if isinstance(node.value, ast.Num) else (
                        node.value.s if isinstance(node.value, ast.Str) else node.value.value)
                    hyperparams[var_name] = var_value
                elif isinstance(node.value, ast.List):
                    var_value = [ast.literal_eval(el) for el in node.value.elts]
                    hyperparams[var_name] = var_value

    return hyperparams

def crawl_project(project_path):
    all_hyperparams = {}
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file == '__init__.py':
                file_path = os.path.join(root, file)
                hyperparams = extract_hyperparams(file_path)
                all_hyperparams.update(hyperparams)

    return all_hyperparams

def save_hyperparams(hyperparams, output_file):
    with open(output_file, 'w') as file:
        json.dump(hyperparams, file, indent=4)
