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

    index_boundary = 'any type except for int'
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


def main():
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
    world_small_df.to_csv('../data/Country_Breakdown.csv')

    
if __name__ == "__main__":
    main()