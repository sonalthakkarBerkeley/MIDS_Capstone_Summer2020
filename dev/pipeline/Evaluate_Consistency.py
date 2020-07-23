import ee
import time
import math
import pandas as pandas

from irrigate30_common import (model_scale, wait_for_task_completion, model_projection, 
                    base_asset_directory, gfsad30_asset_directory,
                    export_asset_table_to_drive, calc_distance, create_bounding_box, source_loc,
                    start_date, end_date, write_image_asset, num_samples, aoi_lat, aoi_lon, aoi_edge_len,
                    band_blue, band_green, band_red, band_nir, get_by_month_data, n_clusters, determine_labels,
                    model_area, AOIs)


def Create_AOI_Box(center_lat,center_lon,edge_len):
    '''
    This function creates the GEE bounding box centered at the coordinate given with the edge length given
    '''
    return ee.Geometry.Rectangle([center_lon-edge_len/2, center_lat-edge_len/2, 
                                                center_lon+edge_len/2, center_lat+edge_len/2])


def Evaluate_Consistency(aoi_lat, aoi_lon, aoi_edge):
    '''
    This function will determine the consistency of output for each pixel in the bounding box
    defined by the cenroid of aoi_lat,aoi_lon with edges of size aoi_edge.
    Consistency is defined as the percentage of times each pixel was labeled the same way.


    Evaluate_Consistency will run nSegments ^ 2 iterations of the model and product an
    Inputs: 
        * Bounded Region to evaluate
            * aoi_lat, aoi_lon, aoi_edge (center point)
    Output:
        * [dataframe, median(% consistent), mean(% consistent)]
        * 1 dataframe:
            * # Positive Cases (Irrigated) [Lat,Lon,# Positive, # Negative, % Consistent]
                Note % Consistent defined as max(% positive, % negative)

    Assumptions:
        * Hardcoding nSegments = 5. Logic would be slightly different for even values. Future
        * Assuming already ran ee.Authenticate(), ee.Initialize()
        * Format of Model_Pipeline dataset will be pandas dataframe with columns: [lat,lon,prediction]

    Note that with 5 Segments the underlying model will be called 25 times (5*5). 
    There will be vertical (latitude shifts) and horizontal (longitude shifts).
    The middle value would be the main area_of_interest while the others correspond to shifts

    '''
    # Assuming already ran ee.Authenticate(), ee.Initialize()
    # Hard-Coded Values (for now): 
    nSegments = 3 # would like this value to remain odd
    aoi_lat, aoi_lon = 43.771114, -116.736866
    aoi_edge_len = 0.005

    area_of_interest_main = Create_AOI_Box(aoi_lat, aoi_lon, aoi_edge_len)

    step_size = edge_len / nSegments
    latitude_vals = [aoi_lat + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]
    longitude_vals = [aoi_lon + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]

    # Build Base Table -- will add to this through iterations
    base_predictions = Model_Pipeline_Placeholder(area_of_interest_main)
    base_predictions['Pos_Case_Count'] = base_predictions['mod']
    base_predictions['Neg_Case_Count'] = 1-base_predictions['mod']
    base_predictions.drop(columns=['mod'], inplace=True)

    for lat in latitude_vals:
        for lon in longitude_vals:
            # Don't rerun base case because we built that to begin with (above)
            if lat == aoi_lat and lon == aoi_lon:
                continue
            else:
                # join new predictions (new lat/lon) in with base predictions and increment counts
                # making assumption that join will work (output from GEE will have consistent Lat/Lon)
                # otherwise may need to round/truncate at some decimal point
                new_predictions = Model_Pipeline_Placeholder(Create_AOI_Box(lat, lon, aoi_edge_len))
                base_predictions = base_predictions.merge(new_predictions, how='left', on=['lat','lon'])
                base_predictions['Pos_Case_Count'] += base_predictions['mod']
                base_predictions['Neg_Case_Count'] += 1-base_predictions['mod']
                base_predictions.drop(columns=['mod'], inplace=True)


    # Now can do calculation of how consistent we have been across the above
    base_predictions['Pct_Consistent'] = base_predictions.apply(lambda row: 
                                            max(row['Pos_Case_Count']/(row['Pos_Case_Count'] + \
                                                          row['Neg_Case_Count']),
                                               row['Neg_Case_Count']/(row['Pos_Case_Count'] + \
                                                                      row['Neg_Case_Count']) )
                                            , axis=1)

    return (base_predictions, 
            base_predictions.Pct_Consistent.median(),
            base_predictions.Pct_Consistent.mean())


def create_overlapping_images(base_lat, base_lon, base_edge, case_num):
    '''
    This function will create the images in GEE
    '''

    nSegments = 3 # would like this value to remain odd

    area_of_interest_base = Create_AOI_Box(base_lat, base_lon, base_edge)

    step_size = base_edge / nSegments
    #(i - (math.floor(nSegments / 2)))
    latitude_vals = [base_lat + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]
    longitude_vals = [base_lon + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]

    # Build Base Table -- will add to this through iterations
    model_area(area_of_interest_base, 'testing/overlap_test_image_base_case_'+str(case_num))

    i = 0
    for lat in latitude_vals:
        for lon in longitude_vals:
            i = i+1
            # Don't rerun base case because we built that to begin with (above)
            if lat == base_lat and lon == base_lon:
                continue
            else:
                print("Creating i=",i)
                model_area(Create_AOI_Box(lat, lon, base_edge), 
                            'testing/overlap_test_image_case_' + str(case_num) + '_' + str(i) )

def evaluate_overlapping_images(nSegments, aoi, case_num):
    base_img = ee.Image(base_asset_directory + "/testing/"+ "overlap_test_image_base_case_" + str(case_num))

    # if there are 5 segments, there are 24 (25-1 that is the base) images
    img_list = ["overlap_test_image_case_" + str(case_num) + '_' +str(i) 
        for i in range(1,nSegments*nSegments+1) if i != math.floor(nSegments * nSegments/2)+1]

    shifted_images = []
    for img in img_list:
        shifted_images.append(ee.Image(base_asset_directory + "/testing/" + img))

    shifted_images.append(base_img)
    IC = ee.ImageCollection(shifted_images)

    count_img = IC.reduce(ee.Reducer.count())
    sum_img = IC.reduce(ee.Reducer.sum())
    # testing this out
    #write_image_asset(count_img, aoi,  "test_count")
    #write_image_asset(sum_img, aoi, "test_sum")
    
    # Now merge these together
    innerJoin = ee.Join.inner()
    filterTimeEq = ee.Filter.equals(leftField= '1', rightField= '1')

    innerJoined = innerJoin.apply(ee.ImageCollection([count_img]), ee.ImageCollection([sum_img]), filterTimeEq)

    joined = innerJoined.map( lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
    joined_img = ee.ImageCollection(joined).max()

    # now create percent similar
    percent_similar = joined_img.expression(
        "(b('class_sum') / b('class_count') > 1 - b('class_sum') / b('class_count') ) ? " +
            "b('class_sum') / b('class_count') : 1-b('class_sum') / b('class_count')"
            ).rename('percent_same');


    #write_image_asset(percent_similar, aoi,  "overlap_case_" + str(case_num) + "_percent_similar")

    # Now reduce the image to get mean and median values
    median_val = percent_similar.reduceRegion(ee.Reducer.median(), geometry = aoi, scale = 30)
    mean_val = percent_similar.reduceRegion(ee.Reducer.mean(), geometry = aoi, scale = 30)
    print("\nMedian val: ", median_val.getInfo())
    print("Mean val :", mean_val.getInfo())
   


def main():

    ee.Initialize()

    # for CASE in range(1,12):
    #     base_aoi_lon = AOIs[CASE][0]
    #     base_aoi_lat = AOIs[CASE][1]
    #     aoi_edge_len = 0.05

    #     base_aoi = create_bounding_box(base_aoi_lat, base_aoi_lon, aoi_edge_len)

    #     # Create images as assets
    #     last_task = create_overlapping_images(base_aoi_lat, base_aoi_lon, aoi_edge_len, CASE)
    # # the above takes some time -- may want to wait until the last task is finished

    # wait for keyboard input -- alert user to make sure the last one has been written
    input("Once all the files are written (check GEE), Press Enter to continue...")

    
    for CASE in range(1,12):
        base_aoi_lon = AOIs[CASE][0]
        base_aoi_lat = AOIs[CASE][1]
        aoi_edge_len = 0.05

        base_aoi = create_bounding_box(base_aoi_lat, base_aoi_lon, aoi_edge_len)
        print('\nCase= ', CASE)
        evaluate_overlapping_images(3, base_aoi, CASE)



if __name__ == '__main__':
    main()

    # note to remove files: example:  earthengine rm users/mbrimmer/w210_irrigated_croplands/overlap_case_6_percent_similar
