import utils.trackers

street_gdf = None
sindex = None

lr_model = None
mr_model = None
depth_model = None

def initialize_gdf(gdf):
    global street_gdf
    global sindex
    street_gdf = gdf
    sindex = street_gdf.sindex

def initialize_vision_models(lr_model_, mr_model_):
    global lr_model
    global mr_model
    lr_model = lr_model_
    mr_model = mr_model_

def initialize_depth_model(model):
    global depth_model
    depth_model = model

def initialize_trackers(log_path):
    global call_counts
    global execution_times
    
    call_counts = utils.trackers.call_counts
    execution_times = utils.trackers.execution_times
    utils.trackers.log_file_path = log_path

