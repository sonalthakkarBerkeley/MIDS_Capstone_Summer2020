# Irrigate30 Model


import ee
import pandas as pd

from irrigate30_common import (model_scale, wait_for_task_completion, model_projection, 
                    base_asset_directory, gfsad30_asset_directory,
                    export_asset_table_to_drive, calc_distance, create_bounding_box, source_loc,
                    start_date, end_date, write_image_asset, num_samples, aoi_lat, aoi_lon, aoi_edge_len,
                    band_blue, band_green, band_red, band_nir, get_by_month_data, n_clusters, determine_labels)


def main():
    ee.Initialize()
    
    aoi = create_bounding_box(aoi_lat, aoi_lon, aoi_edge_len)

    print('The selected area is approximately {:.2f} km by {:.2f} km'.format(\
                calc_distance(aoi_lon-aoi_edge_len/2, aoi_lat, aoi_lon+aoi_edge_len/2, aoi_lat), \
                calc_distance(aoi_lon, aoi_lat-aoi_edge_len/2, aoi_lon, aoi_lat+aoi_edge_len/2)))

    # Create image collection that contains the area of interest
    sat_img_collect = (ee.ImageCollection(source_loc)
                     .filterDate(start_date, end_date)
                     .filterBounds(aoi))


    def calc_NDVI(img):
        ndvi = ee.Image(img.normalizedDifference([band_nir, band_red])).rename(["ndvi"]).copyProperties(img, img.propertyNames())
        composite = img.addBands(ndvi)
        return composite
    
    # SAVI = ((NIR – Red) / (NIR + Red + L)) x (1 + L)
    def calc_SAVI(img):
        """A function to compute Soil Adjusted Vegetation Index."""
        savi =  ee.Image(img.expression(
            '(1 + L) * float(nir - red)/ (nir + red + L)',
            {
                'nir': img.select(band_nir),
                'red': img.select(band_red),
                'L': 0.5
            })).rename(["savi"]).copyProperties(img, img.propertyNames())
        composite = img.addBands(savi)
        return composite

    # EVI = 2.5 * ((NIR – Red) / ((NIR) + (C1 * Red) – (C2 * Blue) + L))
    #     C1=6, C2=7.5, and L=1
    def calc_EVI(img):
        """A function to compute Soil Adjusted Vegetation Index."""
        evi = ee.Image(img.expression(
          '(2.5) * float(nir - red)/ ((nir) + (C1*red) - (C2*blue) + L)',
          {   
              'nir': img.select(band_nir),
              'red': img.select(band_red),
              'blue': img.select(band_blue),
              'L': 0.2,
              'C1': 6,
              'C2': 7.5
          })).rename(["evi"]).copyProperties(img, img.propertyNames())
        composite = img.addBands(evi)
        return composite
    
    def calc_avgVI(img):
        """A function to compute Soil Adjusted Vegetation Index."""
        avgVI = ee.Image(img.expression(
          'float(ndvi + savi + evi)/ 3',
          {   
              'ndvi': img.select('ndvi'),
              'savi': img.select('savi'),
              'evi': img.select('evi')
          })).rename(["avgVI"]).copyProperties(img, img.propertyNames())
        composite = img.addBands(avgVI)
        return composite

    sat_img_collect = sat_img_collect.map(calc_NDVI).map(calc_SAVI).map(calc_EVI).map(calc_avgVI)
    test_img = sat_img_collect.select('B3', 'B2', 'B1', 'ndvi', 'savi').median()

    # Checkpoint 1 -- test writing this to asset to ensure it can be done -- pass (takes ~X minutes)
    # write_image_asset(test_img, "test_asset_description", 'test_img_asset_id')

#     sat_img_collect = sat_img_collect.select('ndvi', 'savi', 'evi')
    sat_img_collect = sat_img_collect.select('avgVI')
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
    joined_img = ee.ImageCollection(joined).median()
    
    # Mask the non-cropland
    # 0 = water, 1 = non-cropland, 2 = cropland, 3 = 'no data'
    cropland = joined_img.select('b1').eq(2)
    joined_img_masked = joined_img.mask(cropland)
    non_cropland = joined_img.select('b1').lt(2) or joined_img.select('b1').gt(2)
    non_cropland = non_cropland.mask(non_cropland)
    

    training = joined_img_masked.sample(region= aoi, scale=model_scale, numPixels=num_samples)

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
        print("\n\nBand Output:", band_output[i])
        temp_df = pd.DataFrame([pd.Series(list(band_output[i].keys())),
                                pd.Series(list(band_output[i].values()))]).T
        temp_df.columns = ['band','mean_value']
        new_name = 'mean_value'+str(i)
        temp_df.rename(columns={'mean_value':new_name}, inplace=True)
        temp_df.set_index('band',inplace=True)
        monthly_df = pd.concat([monthly_df, temp_df], axis=1)
        # monthly_df = monthly_df.merge(temp_df, how='inner', right_index=True, left_index=True)
        # print("Temp_df: ", temp_df)

    print("\n\nMonthly_df: ", monthly_df)

    # TODO -- FOLLOWING CODE ONLY HANDLES 2-CLUSTER CASE. GENERALIZE...
    irrigated_lst = determine_labels(monthly_df)
    if irrigated_lst[0] == 'Not Irrigated':
        # Create Mosaic to view
        mosaic = ee.ImageCollection([
            result.mask(result.select('cluster').eq(0)).visualize({'palette': ['ff0000']}),
            result.mask(result.select('cluster').eq(1)).visualize({'palette': ['00ff00']}),
            non_cropland.visualize({'palette': ['000000']})
        ]).mosaic()
    else:
        mosaic = ee.ImageCollection([
            result.mask(result.select('cluster').eq(1)).visualize({'palette': ['ff0000']}),
            result.mask(result.select('cluster').eq(0)).visualize({'palette': ['00ff00']}),
            non_cropland.visualize({'palette': ['000000']})
        ]).mosaic()


    # Write this to Asset to be viewed later
#     write_image_asset(mosaic, 'blah','blah')

    # export_asset_table_to_drive(f'{base_asset_directory}/samples_{num_samples}')


if __name__ == '__main__':
    main()