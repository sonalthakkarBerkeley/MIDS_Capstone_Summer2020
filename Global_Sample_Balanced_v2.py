import argparse
import ee
import geopandas
import numpy as np
import pandas as pd
from ast import literal_eval
from shapely import wkt
from shapely.geometry import Point, LineString
from shapely.ops import split
import random

def split_country_superseded(country_polygon, max_area):
    '''This works for countries when a single split results in only two areas (not more than two areas)'''
    # Split the country if its area is greater than 250 degree square
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
    
    
def split_country(country_polygon, max_area):
    country_split_lst = [country_polygon]
    
    while True:
#         print('===================')
        index_boundary = next(((i, x) for i, x in enumerate(country_split_lst) if x.area > max_area), 0)
        if isinstance(index_boundary, int):
            break
        country_split_lst.pop(index_boundary[0])
        boundary_split = index_boundary[1]
        centroid = country_polygon.centroid.coords[0]
        boundary_lst = list(boundary_split.exterior.coords)
        # Sometimes splitting is not successful due to weird shape and it is stuck in an infinite loop
        #    Add some randomness to it by allowing the second point to be set anywhere between 1/3 and 2/3
        second_point = round(len(boundary_lst)/2)
        second_point += random.randint(-round(len(boundary_lst)/6), round(len(boundary_lst)/6))
        second_point = min(max(1, second_point), len(boundary_lst)-1)
        split_line = LineString([boundary_lst[0], centroid, boundary_lst[second_point]])
        areas_list = list(split(boundary_split, split_line))
#         for i in areas_list:
#             print(i.area)
        country_split_lst = country_split_lst + areas_list
    return country_split_lst


def generate_random_pixels(polygon_shapely, n_samples, seed_num) -> list:
    # There are in total of 6 categories
    #     0: Non-croplands
    #     1: Croplands: irrigation major
    #     2: Croplands: irrigation minor
    #     3: Croplands: rainfed
    #     4: Croplands: rainfed, minor fragments
    #     5: Croplands: rainfed, rainfed, very minor fragments
    
    if n_samples < 2:
        return []
    else:
        n_samples_per_category = max(round(n_samples/6), 1)
        total_n_samples = round(10*n_samples)
        gfsad_1000 = ee.Image("USGS/GFSAD1000_V1")
        polygon_ee = ee.Geometry.Polygon(list(polygon_shapely.exterior.coords))
        sample_feat_collect = gfsad_1000.sample(numPixels=total_n_samples, projection = "EPSG:3857",
                                region = polygon_ee, geometries = True, scale=30, seed=seed_num)
        sample_df = pd.DataFrame(columns=('coordinate', 'landcover'))
        for pt in sample_feat_collect.getInfo()['features']:
            values_to_add = {'coordinate': pt['geometry']['coordinates'], 'landcover': pt['properties']['landcover']}
            sample_df = sample_df.append(values_to_add, ignore_index=True)
        sample_balanced_df = sample_df.groupby('landcover').head(n_samples_per_category)
        return sample_balanced_df['coordinate'].tolist()


def main(N, max_size, seed_num):
    np.random.seed(seed_num)
    random.seed(seed_num)
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
    print('split starts')
    world_df['geometry_small'] = world_df.apply(lambda row: split_country(row['geometry'], max_size), axis=1)
    print('split completed')
    # Split a list of POLYGONs to multiple POLYGONs, one per row
    world_small_df = world_df.apply(lambda row: pd.Series(row['geometry_small']),axis=1)
    world_small_df['country_code'] = world_df['country_code']
    world_small_df = world_small_df.set_index('country_code').stack().reset_index()
    del world_small_df['level_1']
    world_small_df = world_small_df.rename(columns={'country_code':'country_code', 0:'geometry'})
    # Calculate area
    world_small_df['area'] = world_small_df.apply(lambda row: row['geometry'].area, axis=1)
    world_small_df['samples'] = world_small_df.apply(lambda row: generate_random_pixels(row['geometry'], round(row['area']*N/100), seed_num), axis=1)
    print('sample created')
    world_small_df.to_csv('../data/Global_Samples_Balanced_n{}_m{}_s{}.csv'.format(N, max_size, seed_num))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Sample (lon, lat) globally')
    parser.add_argument('N', type=int, help='The number of pixels sampled is approximately 15000*N/100.')
    parser.add_argument('--max_size', type=int, default=250, help='Specify the max size that is allowed for a given area (in squared degrees) so that GEE limit is not reached.')
    parser.add_argument('--seed_num', type=int, default=210, help='The seed number for splitting and sampling')
    args = parser.parse_args()

    main(args.N, args.max_size, args.seed_num)