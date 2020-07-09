import argparse
import ee
import geopandas
import numpy as np
import pandas as pd
from ast import literal_eval
from shapely import wkt
from shapely.geometry import Point, LineString

def gfsad30_coords(center_lat, center_lon, edge_len):
    '''This works for generating coordinates within the GFSAD30 dataset'''
    coord_in_gfsad30_cropland = False
    print("center_lat, center_lon: ", center_lat, center_lon)
    area_of_interest = ee.Geometry.Rectangle([center_lon-edge_len/2, center_lat-edge_len/2, center_lon+edge_len/2, center_lat+edge_len/2])                                       # 10174268870
    gfsad30_asset = 'users/ajsohn/GFSAD30';
    gfsad_30_IC = ee.ImageCollection(gfsad30_asset).filterBounds(area_of_interest)
    img_calc_month_dict = gfsad_30_IC.median()
    img_calc_month2 = img_calc_month_dict.addBands(ee.Image.pixelLonLat())
    # print("img_calc_month2")
    pixelsDict = img_calc_month2.reduceRegion(reducer=ee.Reducer.toList(), geometry=area_of_interest, maxPixels=1e13, scale=300)
    try:
        b1_series = pd.Series(np.array((ee.Array(pixelsDict.get("b1")).getInfo())), name='b1')
        print("b1_series", b1_series.max())
        if round(b1_series.max()) == 2:
            coord_in_gfsad30_cropland = True
            print("coord_in_gfsad30_cropland: ", coord_in_gfsad30_cropland)
    except:
        warnings.warn('Missing satellite data.')

    return coord_in_gfsad30_cropland

def generate_boundaries(polygon_shapely) -> list:
    '''This works for generating max min latitudes and longitudes for a country shape'''
    min_lon, min_lat, max_lon, max_lat = (polygon_shapely.bounds)
    return [min_lon, min_lat, max_lon, max_lat]

def generate_edge_coords(polygon_shapely, country_code, edge_len) -> list:
    '''This works for generating rectangle coords for a country shape'''
    return_list_coords = []
    print(country_code)
    world_gdf = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world_gdf.crs = "EPSG:3857"
    return_list_coords = []
    min_lon, min_lat, max_lon, max_lat = polygon_shapely.bounds
    dict_coords = {}
    point_list = []
    lat_list = []
    lon_list = []
    counter = 0
    center_lon = min_lon
    center_lat = max_lat
    while center_lon < (max_lon):
        print("outside while:", center_lat, center_lon)
        while center_lat > (min_lat):
            print("inside while:", center_lat, center_lon)
            if (gfsad30_coords(center_lat, center_lon, edge_len)):
                print("counter: ", counter)
                point_list.append(counter)
                lat_list.append(center_lat)
                lon_list.append(center_lon)
            counter +=1
            center_lat -= edge_len

        center_lon += edge_len
        center_lat = max_lat
        print("center_lat, center_lon inside coords: ", center_lat, center_lon)

    dict_coords['POINTS'] = point_list
    dict_coords['Latitude'] = lat_list
    dict_coords['Longitude'] = lon_list
    print("Total coords: ", counter)
    df = pd.DataFrame(dict_coords)
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
    gdf.crs = "EPSG:3857"
    point_within_country = geopandas.sjoin(gdf, world_gdf[world_gdf.iso_a3 == country_code], how="inner", op='intersects')
    for index, row in point_within_country.iterrows():
      (lat, lon) = (row['Latitude'], row['Longitude'])

      return_list_coords.append((lat, lon))
    
    print("return_list_coords: ", len(return_list_coords))
    return return_list_coords

def main(C_ID, edge_len):
    
    ee.Initialize()
    
    world_df = pd.read_csv('../data/Global_Samples_Balanced_n200_m50_s213.csv', index_col=[0])
    world_df.head()
    world_df['geometry'] = world_df['geometry'].apply(wkt.loads)
    world_gdf = geopandas.GeoDataFrame(world_df, geometry='geometry')
    
    world_gdf = world_gdf.reset_index()
    
    # Add coundaries
    world_gdf['boundaries'] = world_gdf.apply(lambda row: generate_boundaries(row['geometry']), axis=1)
    # Add coords
    print(world_gdf.iloc[C_ID])
#     world_gdf.iloc[617]['coords'] = world_gdf.iloc[617].apply(generate_edge_coords(world_gdf.iloc[617]['geometry'], world_gdf.iloc[617]['country_code'], edge_len))
    world_gdf['coords'] = world_gdf.apply(lambda x: [], axis=1)
    test_val = []
    test_val = generate_edge_coords(world_gdf.iloc[C_ID]['geometry'], world_gdf.iloc[C_ID]['country_code'], edge_len)
    print("test_val: ", test_val)
#     world_gdf.iloc[C_ID]['coords'] = test_val
    world_gdf.at[C_ID, 'coords'] = test_val
    #(lambda row: generate_edge_coords(row['geometry'], row['country_code'], row.index, edge_len), axis=1)
    
    world_gdf.to_csv('../data/Global_Balanced_GFSAD30.csv')


    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'GFSAD30 croplands (lon, lat) globally')
    parser.add_argument('C_ID', type=int, help='The country index.')
    parser.add_argument('edge_len', type=float, help='The coordinate list based on edge_len for a shape.')
    args = parser.parse_args()

    main(args.C_ID, args.edge_len)