from google.cloud import storage
from google.cloud import bigquery

import hashlib
import hmac
import base64
import urllib.parse as urlparse
import requests

from io import BytesIO
from PIL import Image, ExifTags

import ast
import re

import numpy as np
from torchvision.transforms import ToTensor

from env_vars import SERVICE_ACCOUNT_JSON_PATH, API_URL_SECRET
from utils.trackers import time_tracker, call_counter

import logging
logger = logging.getLogger()

def init_clients(service_account_json_path):
  storage_client = storage.Client.from_service_account_json(service_account_json_path)
  bigquery_client = bigquery.Client.from_service_account_json(service_account_json_path)
  return storage_client, bigquery_client

storage_client, bigquery_client = init_clients(SERVICE_ACCOUNT_JSON_PATH)

def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()

def convert_location_string_to_dict(input_dict):
    if 'location' in input_dict and isinstance(input_dict['location'], str):
        try:
            # Convert the string representation to an actual dictionary
            input_dict['location'] = ast.literal_eval(input_dict['location'])
        except ValueError:
            print("Error: The 'location' string is not a valid dictionary representation.")
    return input_dict

@time_tracker
def upload_image(image_bytes, bucket_name, blob_name, metadata=None):
    """Uploads an image to the specified GCS bucket with metadata."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # Set metadata if provided
    if metadata:
        blob.metadata = metadata
    blob.upload_from_file(image_bytes, content_type='image/jpeg')

def generate_streetview_url(lat, lon, api_key, size="640x640", fov=120, pitch=0, heading=None):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    if (heading) or heading == 0:
        params = f"size={size}&location={lat},{lon}&fov={fov}&pitch={pitch}&heading={heading}&return_error_code=true&source=outdoor&key={api_key}"
    else:
        params = f"size={size}&location={lat},{lon}&fov={fov}&pitch={pitch}&return_error_code=true&source=outdoor&key={api_key}"
    return f"{base_url}?{params}"

def generate_streetview_metadata_url(lat, lon, api_key):
    base_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = f"location={lat},{lon}&key={api_key}"
    parameterized_url = f"{base_url}?{params}"
    signed_url = sign_url(parameterized_url, API_URL_SECRET)
    return signed_url

@time_tracker
def request_and_parse_streetview_metadata(lat, lon, api_key):
    metadata_url = generate_streetview_metadata_url(lat, lon, api_key)
    response = requests.get(metadata_url)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: Received status code {response.status_code}"

@time_tracker
def download_image_if_exists(bucket_name, source_blob_name):
    """Checks if a blob exists in the bucket and downloads it if it does."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if blob.exists():
        image_data = BytesIO()
        blob.download_to_file(image_data)
        image_data.seek(0)  # Rewind the buffer to the beginning

        image = Image.open(image_data)
        # print('Image Retrieved')
        return image
    else:
        # print('Image Fetched From API')
        return False

def get_blob_metadata(bucket_name, blob_name):
    """Fetches the metadata of a blob from the specified GCS bucket."""

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    # Fetch the blob metadata
    blob.reload()  # Ensures the metadata is up-to-date
    return blob.metadata

def enrich_metadata(metadata, size, fov, pitch, heading):
    # Add custom metadata
    custom_metadata = {
        'size': size,
        'fov': fov,
        'pitch': pitch,
        'heading':heading
    }
    metadata.update(custom_metadata)
    return metadata

def create_filename_from_params(lat, lon, fov, pitch, heading):
    filename = 'image_' + str(lat) + '_' + str(lon) + '_fov=' + str(fov) + '_pitch=' + str(pitch) + '_heading=' + str(heading) + '.png'
    return filename

@call_counter
@time_tracker
def request_streetview_image(image_url):
    return requests.get(image_url)

def fetch_and_save_image_with_metadata(lat, lon, api_key, bucket_name, bucket_path, filename, size="640x640", fov=120, pitch=0, heading=None, force=False):
    image_exists = download_image_if_exists(bucket_name, bucket_path + filename)

    if image_exists and not force:
      metadata = get_blob_metadata(bucket_name, bucket_path + filename)
      metadata = convert_location_string_to_dict(metadata)
      return image_exists, metadata

    # Generate URLs
    image_url = generate_streetview_url(lat, lon, api_key, size, fov, pitch, heading)
    metadata = request_and_parse_streetview_metadata(lat, lon, api_key)
    metadata = enrich_metadata(metadata, size, fov, pitch, heading)

    response = request_streetview_image(image_url)

    if response.status_code == 200:
        # Open the image and add metadata as a comment
        image = Image.open(BytesIO(response.content))
        image.info['comment'] = str(metadata)

        # Save image with metadata to BytesIO object
        image_bytes = BytesIO()
        image.save(image_bytes, format=image.format)
        image_bytes.seek(0)

        # Upload to GCS
        upload_image(image_bytes, bucket_name, bucket_path + filename, metadata=metadata)
    else:
        # print(f"Error: Failed to fetch image, status code {response.status_code}")
        logger.info(f"Error: Failed to fetch image, status code {response.status_code}")
        return None, None

    return image, metadata

def read_metadata(image_path):
    # Load the image
    img = Image.open(image_path)

    # Attempt to extract EXIF data
    exif_data = img._getexif()

    # If the image contains EXIF data, convert it to a human-readable format
    if exif_data:
        readable_exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}
        return readable_exif
    else:
        return "No EXIF data found."

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = ToTensor()(image)
    return image

def parse_image_filename(filename):
    # Regular expression to match the pattern
    pattern = r'image_([-\d\.]+)_([-\d\.]+)_fov=(\d+)_pitch=([-\d\.]+)_heading=([\w\.]+)\.png'

    # Search for the pattern in the filename
    match = re.search(pattern, filename)

    if match:
        lat = np.round(float(match.group(1)), 6)
        lon = np.round(float(match.group(2)), 6)
        fov = int(match.group(3))
        pitch = float(match.group(4))

        # Heading might be 'None'
        heading_str = match.group(5)
        heading = float(heading_str) if heading_str != 'None' else 0

        return lat, lon, fov, pitch, heading
    else:
        raise ValueError("Filename does not match the expected format. Filename was: " + filename)

def save_image_to_dataset(dataset_name, image, metadata, bucket_path, bucket_name, filename, image_type, caller):
    datasets_path = '/'.join(bucket_path.split('/')[:-2]) + dataset_name
    if caller == 'walk':
        location_path = '/' + str(caller.images[0].location.lat) + '_' + str(caller.images[0].location.lon) + '/'
        walk_id = 'walk_' + str(len(caller.images[0].walks)) + '/'
        starter_folder = caller.images[0].filename[:caller.images[0].filename.rfind('.')] + '/'
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

        # print(datasets_path + filename_with_type)
        upload_image(image_bytes, bucket_name, datasets_path + filename_with_type, metadata=metadata)