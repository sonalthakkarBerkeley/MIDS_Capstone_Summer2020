# Irrigate30 Model


import ee
import pandas as pd

from irrigate30_common import (model_scale, wait_for_task_completion, model_projection, 
                    base_asset_directory, gfsad30_asset_directory,
                    export_asset_table_to_drive, calc_distance, create_bounding_box, source_loc,
                    start_date, end_date, write_image_asset, num_samples, aoi_lat, aoi_lon, aoi_edge_len,
                    band_blue, band_green, band_red, band_nir, get_by_month_data, n_clusters, determine_labels,
                    model_area)


def main():
    ee.Initialize()
    
    aoi = create_bounding_box(aoi_lat, aoi_lon, aoi_edge_len)

    print('The selected area is approximately {:.2f} km by {:.2f} km'.format(\
                calc_distance(aoi_lon-aoi_edge_len/2, aoi_lat, aoi_lon+aoi_edge_len/2, aoi_lat), \
                calc_distance(aoi_lon, aoi_lat-aoi_edge_len/2, aoi_lon, aoi_lat+aoi_edge_len/2)))

    model_area(aoi, "Name_edge_1")

    


if __name__ == '__main__':
    main()