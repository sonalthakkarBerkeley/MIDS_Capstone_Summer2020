# Irrigate30 Model


import ee

from irrigate30_common import (model_scale, wait_for_task_completion, model_projection, 
                    base_asset_directory, gfsad30_asset_directory,
                    export_asset_table_to_drive, calc_distance, create_bounding_box, source_loc,
                    start_date, end_date, write_image_asset, num_samples, aoi_lat, aoi_lon, aoi_edge_len,
                    band_blue, band_green, band_red, band_nir, get_by_month_data)


def main():
    ee.Initialize()

    print('got here')
    
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
    print("\n\nCheckpoint 4 -- does this look right?", innerJoined.getInfo())

    joined = innerJoined.map( lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
    print("\n\nCheckpoint 5 -- does this look right?", joined.getInfo())

    joined_img = ee.ImageCollection(joined).median()
    # export_asset_table_to_drive(f'{base_asset_directory}/samples_{num_samples}')


if __name__ == '__main__':
    main()