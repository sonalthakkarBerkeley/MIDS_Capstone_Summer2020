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
model_snapshot_version = ""

model_snapshot_path_prefix = base_asset_directory + "/" + model_snapshot_version

num_samples = 5000
aoi_lat = 43.771114
aoi_lon = -116.736866
aoi_edge_len = 0.05

AOIs = [
	[0,0], 						# EMPTY
	[82.121452, 21.706688],     # C01 (India): 
    [-94.46643, 48.76297],      # C02 (Canada):  
    [-105.69772, 49.316722],    # C03 (Canada): 
    [10.640815, 52.185072],     # C04 (Germany): strange result... look at S2 layer! clouds?
    [10.7584699, 52.2058339],   # C05 (Germany): strange result
    [33.857852, 46.539389],     # C06 (Ukraine): good example, get second opinion
    [36.58565, 47.0838],        # C07 (Ukraine): Looks like color-labels are backwards. Notice how S2 is different from "Satellite" !!! 
    [38.34523, 30.22176],       # C08 (Saudi Arabia): Looks great, start with this!
    [-64.075199, -31.950112],   # C09 (Argentina): NEW
    [67.359826, 43.55412],      # C10 (Uzbekistan): mostly good
    [-46.2607, -11.93067] 		# C11 (Brazil): strange result
]

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


def export_asset_table_to_drive(asset_id, wait=False):
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
    if wait:
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

def write_image_asset(image, region, image_asset_id, wait=False):
    image_asset_id = model_snapshot_path_prefix  +  image_asset_id

    task = ee.batch.Export.image.toAsset(
        crs=model_projection,
        region=region,
        image=image,
        scale=model_scale,
        assetId=image_asset_id,
        maxPixels=1e13
    )
    task.start()
    if wait:
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

	return ee.Image(merged).select(['ndvi','ndvi_1','ndvi_2','ndvi_3','ndvi_4','ndvi_5','ndvi_6', 'ndvi_7','ndvi_8','ndvi_9','ndvi_10','ndvi_11'],
		['ndvi01','ndvi02','ndvi03','ndvi04','ndvi05','ndvi06','ndvi07', 'ndvi08','ndvi09','ndvi10','ndvi11','ndvi12'])

def determine_labels(df):
	# Heuristic 1:
	col_0, col_1 = df.mean()
	print("Mean: ", df.mean())
	if col_0 > col_1:
		return ["Irrigated", "Not Irrigated"]
	else:
		return ["Not Irrigated", "Irrigated"]


def model_area(aoi, output_img_name):
	# Create image collection that contains the area of interest
    sat_img_collect = (ee.ImageCollection(source_loc)
                     .filterDate(start_date, end_date)
                     .filterBounds(aoi))


    def calc_NDVI(img):
        ndvi = ee.Image(img.normalizedDifference([band_nir, band_red])).rename(["ndvi"]).copyProperties(img, img.propertyNames())
        composite = img.addBands(ndvi)
        return composite

    sat_img_collect = sat_img_collect.map(calc_NDVI)
    test_img = sat_img_collect.select('B3', 'B2', 'B1', 'ndvi').median()

    # Checkpoint 1 -- test writing this to asset to ensure it can be done -- pass (takes ~X minutes)
    # write_image_asset(test_img, "test_asset_description", 'test_img_asset_id')

    sat_img_collect = sat_img_collect.select('ndvi')
    # ---------- GET MONTHLY DATA ---------
    byMonth = get_by_month_data(sat_img_collect)


    # # Get GFSAD30
    GFSAD30_IC = ee.ImageCollection(gfsad30_asset_directory).filterBounds(aoi)
    # # Convert to image and back to get rid of extra images in collection
    GFSAD30_IC = ee.ImageCollection( GFSAD30_IC.max() )

    # # Checkpoint 3 -- does this look right for GFSAD?
    # print("\n\nCheckpoint 3 -- does GFSAD30_IC look right?", GFSAD30_IC.getInfo())

    # Define an inner join
    innerJoin = ee.Join.inner()
    filterTimeEq = ee.Filter.equals(leftField= '1',rightField= '1')

    innerJoined = innerJoin.apply(ee.ImageCollection([byMonth]), GFSAD30_IC, filterTimeEq);
    # # Checkpoint 4 -- does this look right?
    # print("\n\nCheckpoint 4 -- does this look right?", innerJoined.getInfo())

    joined = innerJoined.map( lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
    # print("\n\nCheckpoint 5 -- does this look right?", joined.getInfo())

    # Now turn it into single image
    joined_img = ee.ImageCollection(joined).max()
    
    # Mask the non-cropland
    # 0 = water, 1 = non-cropland, 2 = cropland, 3 = 'no data'
    cropland = joined_img.select('b1').eq(2)
    joined_img_masked = joined_img.mask(cropland)
    non_cropland = joined_img.select('b1').lt(2) or joined_img.select('b1').gt(2)
    non_cropland = non_cropland.mask(non_cropland)
    
    # pre-changes line
    #training = joined_img_masked.sample(region= aoi, scale=model_scale, numPixels=num_samples)
    # stratified line
    training = joined_img.cast({'b1':"int8"},['b1', 'ndvi01','ndvi02','ndvi03','ndvi04','ndvi05','ndvi06','ndvi07', 'ndvi08','ndvi09','ndvi10','ndvi11','ndvi12'])\
        .stratifiedSample(region= aoi, classBand = 'b1', numPoints = num_samples,
            classValues = [0, 1, 3], 
            classPoints = [0, 0, 0],
            scale=model_scale)

    # Instantiate the clusterer and train it.
    clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)

    # Cluster the input using the trained clusterer.
    result = joined_img_masked.cluster(clusterer)
    # print("\n\nResult:",result.getInfo())

    # loop through all the clusters
    band_output = []
    monthly_df = pd.DataFrame()
    for i in range(n_clusters):
        # Create Aggregate NDVI Signature
        band_output.append(joined_img_masked.mask(result.select('cluster').eq(i) ) \
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=aoi, maxPixels=1e13, scale=model_scale) \
            .getInfo())
        # print("\n\nBand Output:", band_output[i])
        temp_df = pd.DataFrame([pd.Series(list(band_output[i].keys())),
                                pd.Series(list(band_output[i].values()))]).T
        temp_df.columns = ['band','mean_value']
        new_name = 'mean_value'+str(i)
        temp_df.rename(columns={'mean_value':new_name}, inplace=True)
        temp_df.set_index('band',inplace=True)
        monthly_df = pd.concat([monthly_df, temp_df], axis=1)
        # monthly_df = monthly_df.merge(temp_df, how='inner', right_index=True, left_index=True)
        # print("Temp_df: ", temp_df)

    # print("\n\nMonthly_df: ", monthly_df)

    # TODO -- FOLLOWING CODE ONLY HANDLES 2-CLUSTER CASE. GENERALIZE...
    irrigated_lst = determine_labels(monthly_df)
    
    # Now 
    res_w_gfsad30_temp = innerJoin.apply(ee.ImageCollection([result]), GFSAD30_IC, filterTimeEq);
    res_w_gfsad30 = res_w_gfsad30_temp.map( lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
    res_w_gfsad30_img = ee.ImageCollection(res_w_gfsad30).median()

    # Create single band image with results and gfsad
    if irrigated_lst[0] == 'Not Irrigated':
        singleBand = res_w_gfsad30_img.expression(
        "(b('b1') != 2) ? 0 " +
            ": (b('cluster') == 0) ? 0 : 1"
            ).rename('class');
    else:
        singleBand = res_w_gfsad30_img.expression(
        "(b('b1') != 2) ? 0 " +
            ": (b('cluster') == 1) ? 0 : 1"
            ).rename('class');

    # print("\n\nResult singleBand:",singleBand.getInfo())
    # export_asset_table_to_drive(f'{base_asset_directory}/samples_{num_samples}')    




    # print("\n\nmosaic: ", ee.ImageCollection(mosaic).getInfo())
    # Write this to Asset to be viewed later
   
    write_image_asset(singleBand, aoi, output_img_name)

    # export_asset_table_to_drive(f'{base_asset_directory}/samples_{num_samples}')
