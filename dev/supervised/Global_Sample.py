import argparse
import geopandas
import numpy as np
import pandas as pd
from ast import literal_eval
from shapely import wkt
from shapely.geometry import Point

def generate_random_pixels(polygon, n_samples) -> list:
    rand_pixel_lst = []
    for num in range(n_samples):
        within = False
        while not within:
            min_lon, min_lat, max_lon, max_lat = polygon.bounds
            lon = np.random.uniform(min_lon, max_lon)
            lat = np.random.uniform(min_lat, max_lat)
            within = polygon.contains(Point(lon,lat))
        rand_pixel_lst.append((lon, lat))
    return rand_pixel_lst

def main(N):
    np.random.seed(210)
    
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
    # Convert back to GeoDataFrame
    world_gdf = geopandas.GeoDataFrame(world_df, geometry='geometry')
    world_gdf["area"] = world_gdf.apply(lambda row: row["geometry"].area, axis=1)
    world_gdf["samples"] = world_gdf.apply(lambda row: generate_random_pixels(row["geometry"], round(row['area']*N/100)), axis=1)
    pd.DataFrame(world_gdf).to_csv('../data/Global_samples_{}.csv'.format(N))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Sample (lon, lat) globally')
    parser.add_argument('N', type=int, help='The number of pixels sampled is approximately 15000*N/100.')
    args = parser.parse_args()

    main(args.N)