# Irrigate30 Common Functions and Definitions


import time
import itertools
import datetime
import json
import pandas as pd

from math import sin, cos, sqrt, atan2, radians

import ee

model_scale = 30
n_clusters = 2

# gfsad30_asset_directory is where GFSAD30 is located
gfsad30_asset_directory = "users/ajsohn/GFSAD30"

# base_asset_directory is where we are going to store output images
base_asset_directory = "users/mbrimmer/w210_irrigated_croplands"

model_projection = "EPSG:4326"
model_snapshot_version = "testing_v1"

model_snapshot_path_prefix = base_asset_directory + "/" + model_snapshot_version

num_samples = 50000
aoi_lat = 21.706688 #43.771114
aoi_lon = 82.121452 #-116.736866
aoi_edge_len = 4.0 #0.1 #05

source_loc = 'COPERNICUS/S2'
start_date = '2018-1-01'
end_date = '2018-12-31'

band_blue = 'B2' #10m
band_green = 'B3' #10m
band_red = "B4"  #10m
band_nir = 'B8'  #10m

def wait_for_task_completion(tasks, exit_if_failures=False):
    sleep_time = 10  # seconds
    done = False
    failed_tasks = []
    while not done:
        failed = 0
        completed = 0
        for t in tasks:
            status = t.status()
            print(f"{status['description']}: {status['state']}")
            if status['state'] == 'COMPLETED':
                completed += 1
            elif status['state'] in ['FAILED', 'CANCELLED']:
                failed += 1
                failed_tasks.append(status)
        if completed + failed == len(tasks):
            print(f"All tasks processed in batch: {completed} completed, {failed} failed")
            done = True
        time.sleep(sleep_time)
    if failed_tasks:
        print("--- Summary: following tasks failed ---")
        for status in failed_tasks:
            print(status)
        print("--- END Summary ---")
        if exit_if_failures:
            raise NotImplementedError("There were some failed tasks, see report above")


def export_asset_table_to_drive(asset_id):
    fc = ee.FeatureCollection(asset_id)
    folder = asset_id.replace('/', '_')
    max_len = 100
    if len(folder) >= max_len:
        folder = folder[:max_len]
        print(f"folder length is too long (truncating to {max_len})")
    print(f"Downloading table {asset_id} to gdrive: {folder}")
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        folder=folder,
        description=folder,
        fileFormat='GeoJSON'
    )
    task.start()
    wait_for_task_completion([task], True)

def create_bounding_box(aoi_lat, aoi_lon, aoi_edge_len):
	return ee.Geometry.Rectangle([aoi_lon-aoi_edge_len/2, aoi_lat-aoi_edge_len/2, 
								aoi_lon+aoi_edge_len/2, aoi_lat+aoi_edge_len/2])

def calc_distance(lon1, lat1, lon2, lat2):
        # Reference: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
        # approximate radius of earth in km
        R = 6373.0
        lon1 = radians(lon1)
        lat1 = radians(lat1)
        lon2 = radians(lon2)
        lat2 = radians(lat2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

def write_image_asset(image, asset_description, image_asset_id):
    asset_description = 'test_image'
    image_asset_id = model_snapshot_path_prefix + asset_description +  '_image'

    task = ee.batch.Export.image.toAsset(
        crs=model_projection,
        image=image,
        scale=model_scale,
        assetId=image_asset_id,
        description=asset_description,
        maxPixels=1e13
    )
    task.start()
    wait_for_task_completion([task], exit_if_failures=True)

def get_by_month_data(img):
	months = ee.List.sequence(1,12)
	byMonth = ee.ImageCollection.fromImages(
  		months.map(lambda m: img.filter(ee.Filter.calendarRange(m, m, 'month')).median().set('month', m)
  			).flatten()
	)

	# Take all the satellite bands that have been split into months 
	# as different images in collection (byMonth), and merge into different bands
	def mergeBands(image, previous):
		return ee.Image(previous).addBands(image).copyProperties(image, image.propertyNames())

	merged = byMonth.iterate(mergeBands, ee.Image())
	# Checkpoint -- does this look right?
	# print("byMonth:", byMonth.getInfo())

# 	return ee.Image(merged).select(['ndvi','ndvi_1','ndvi_2','ndvi_3','ndvi_4','ndvi_5','ndvi_6', 'ndvi_7','ndvi_8','ndvi_9','ndvi_10','ndvi_11','savi','savi_1','savi_2','savi_3','savi_4','savi_5','savi_6', 'savi_7','savi_8','savi_9','savi_10','savi_11','evi','evi_1','evi_2','evi_3','evi_4','evi_5','evi_6', 'evi_7','evi_8','evi_9','evi_10','evi_11'],
# 		['ndvi01','ndvi02','ndvi03','ndvi04','ndvi05','ndvi06','ndvi07', 'ndvi08','ndvi09','ndvi10','ndvi11','ndvi12','savi01','savi02','savi03','savi04','savi05','savi06','savi07', 'savi08','savi09','savi10','savi11','savi12','evi01','evi02','evi03','evi04','evi05','evi06','evi07', 'evi08','evi09','evi10','evi11','evi12'])
	return ee.Image(merged).select(['avgVI','avgVI_1','avgVI_2','avgVI_3','avgVI_4','avgVI_5','avgVI_6', 'avgVI_7','avgVI_8','avgVI_9','avgVI_10','avgVI_11'],
		['avgVI01','avgVI02','avgVI03','avgVI04','avgVI05','avgVI06','avgVI07', 'avgVI08','avgVI09','avgVI10','avgVI11','avgVI12'])

def determine_labels(df):
	return ["Not Irrigated", "Irrigated"]