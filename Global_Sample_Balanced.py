import argparse
import ee
import geopandas
import numpy as np
import pandas as pd
from ast import literal_eval
from shapely import wkt
from shapely.geometry import Point, LineString
from shapely.ops import split

def split_country_superseded(country_polygon):
    '''This works for countries when a single split results in only two areas (not more than two areas)'''
    # Split the country if its area is greater than 250 degree square
    max_area = 250
    n_splits = math.ceil(country_polygon.area / max_area)
    if n_splits == 1:
        return country_polygon
    else:
        centroid = country_polygon.centroid.coords[0]
        boundary_points = [pt for i, pt in enumerate(country_polygon.exterior.coords) if i % math.ceil(len(country_polygon.exterior.coords) / n_splits) == 0]

        # original selected boundary points
        boundary_points_org = boundary_points.copy()
        # Roll the list to the left
        boundary_points.append(boundary_points.pop(0))
        line_dict = dict()
        for i, (a, b) in enumerate(zip(boundary_points_org, boundary_points)):
            line_dict[i] = LineString([a, centroid, b])
        # Create a dictionary where each value is a segment of the country
        country_segment_dict = dict()
        remaining = country_polygon
        for s in range(n_splits-1):
            area_area = split(remaining, line_dict[s])
            country_segment_dict[s] = area_area[0]
            remaining_iter = iter(area_area)
            next(remaining_iter)
            remaining = next(remaining_iter)
        country_segment_dict[n_splits-1] = remaining
        return list(country_segment_dict.values())
    
    
def split_country(country_polygon):
    max_area = 250
    country_split_lst = [country_polygon]

    while True:
        index_boundary = next(((i, x) for i, x in enumerate(country_split_lst) if x.area > max_area), 0)
        if isinstance(index_boundary, int):
            break
        country_split_lst.pop(index_boundary[0])
        boundary_split = index_boundary[1]
        centroid = country_polygon.centroid.coords[0]
        boundary_lst = list(boundary_split.exterior.coords)
        split_line = LineString([boundary_lst[0], centroid, boundary_lst[round(len(boundary_lst)/2)]])
        areas_list = list(split(boundary_split, split_line))
        country_split_lst = country_split_lst + areas_list
    return country_split_lst

def generate_boundaries(polygon_shapely) -> list:
    '''This works for generating max min latitudes and longitudes for a country shape'''
    min_lon, min_lat, max_lon, max_lat = (polygon_shapely.bounds)
    return [min_lon, min_lat, max_lon, max_lat]

def generate_edge_coords(polygon_shapely, country_code, edge_len) -> list:
    '''This works for generating rectangle coords for a country shape'''
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
        while center_lat > (min_lat):
            # print(center_lat, center_lon)
            point_list.append(counter)
            lat_list.append(center_lat)
            lon_list.append(center_lon)
            counter +=1
            center_lat -= edge_len
        
        center_lon += edge_len
        center_lat = max_lat
        
    dict_coords['POINTS'] = point_list
    dict_coords['Latitude'] = lat_list
    dict_coords['Longitude'] = lon_list
    # print("Total coords: ", counter)
    df = pd.DataFrame(dict_coords)
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
    gdf.crs = "EPSG:3857"
    point_within_country = geopandas.sjoin(gdf, world_gdf[world_gdf.iso_a3 == country_code], how="inner", op='intersects')
    for index, row in point_within_country.iterrows():
      (lat, lon) = (row['Latitude'], row['Longitude'])
    
      return_list_coords.append((lat, lon))
    
    return return_list_coords

def generate_random_pixels(polygon_shapely, n_samples) -> list:
    # There are in total of 6 categories
    #     0: Non-croplands
    #     1: Croplands: irrigation major
    #     2: Croplands: irrigation minor
    #     3: Croplands: rainfed
    #     4: Croplands: rainfed, minor fragments
    #     5: Croplands: rainfed, rainfed, very minor fragments
    
    # fixme: if n_samples is less than 6, we select no samples
    n_samples_per_category = round(n_samples/6)
    total_n_samples = round(10*n_samples)
    gfsad_1000 = ee.Image("USGS/GFSAD1000_V1")
    polygon_ee = ee.Geometry.Polygon(list(polygon_shapely.exterior.coords))
    sample_feat_collect = gfsad_1000.sample(numPixels=total_n_samples, projection = "EPSG:3857",
                            region = polygon_ee, geometries = True, scale=30, seed=210)
    sample_df = pd.DataFrame(columns=('coordinate', 'landcover'))
    for pt in sample_feat_collect.getInfo()['features']:
        values_to_add = {'coordinate': pt['geometry']['coordinates'], 'landcover': pt['properties']['landcover']}
        sample_df = sample_df.append(values_to_add, ignore_index=True)
    sample_balanced_df = sample_df.groupby('landcover').head(n_samples_per_category)
    return sample_balanced_df['coordinate'].tolist()


def main(N, edge_len):
    np.random.seed(210)
    ee.Initialize()
    
    world_gdf = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # Remove -99 countries, Antarctica and Seven seas (Fr. S. Antarctic Lands)
    world_gdf = world_gdf[(world_gdf['iso_a3']!='-99') & (world_gdf['continent']!='Antarctica') & (world_gdf['continent']!='Seven seas (open ocean)')]
    # Split MULTIPOLYGON into POLYGONs
    world_df = pd.DataFrame(world_gdf)
    world_df = world_gdf.apply(lambda row: pd.Series(row['geometry']),axis=1)
    world_df['iso_a3'] = world_gdf['iso_a3']
    world_df = world_df.set_index('iso_a3').stack().reset_index()
    del world_df['level_1']
    # Rename columns
    world_df = world_df.rename(columns={'iso_a3':'country_code', 0:'geometry'})
    # Split the country if it's too big
    world_df['geometry_small'] = world_df.apply(lambda row: split_country(row['geometry']), axis=1)
    # Split a list of POLYGONs to multiple POLYGONs, one per row
    world_small_df = world_df.apply(lambda row: pd.Series(row['geometry_small']),axis=1)
    world_small_df['country_code'] = world_df['country_code']
    world_small_df = world_small_df.set_index('country_code').stack().reset_index()
    del world_small_df['level_1']
    world_small_df = world_small_df.rename(columns={'country_code':'country_code', 0:'geometry'})
    # Calculate area
    world_small_df['area'] = world_small_df.apply(lambda row: row['geometry'].area, axis=1)
    world_small_df['samples'] = world_small_df.apply(lambda row: generate_random_pixels(row['geometry'], round(row['area']*N/100)), axis=1)
    # Add coundaries
    world_small_df['boundaries'] = world_small_df.apply(lambda row: generate_boundaries(row['geometry']), axis=1)
    # Add coords
    world_small_df['coords'] = world_small_df.apply(lambda row: generate_edge_coords(row['geometry'], row['country_code'],edge_len), axis=1)
    
    world_small_df.to_csv('../data/Global_samples_Balanced_{}.csv'.format(N))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Sample (lon, lat) globally')
    parser.add_argument('N', type=int, help='The number of pixels sampled is approximately 15000*N/100.')
    parser.add_argument('edge_len', type=float, help='The coordinate list based on edge_len for a shape.')
    args = parser.parse_args()

    main(args.N, args.edge_len)