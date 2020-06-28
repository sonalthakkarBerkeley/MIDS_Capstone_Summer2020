import ee
import geopandas
import datetime
import numpy as np
import pandas as pd
from ast import literal_eval
from shapely import wkt
from shapely.geometry import Point
import warnings
import argparse
import time

def main(country, year, resolution=30):
    ee.Initialize()
                
    def last_day_of_month(any_day):
        next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
        return next_month - datetime.timedelta(days=next_month.day)

    def calc_YYYYMM(img):
        return img.set('YYYYMM', img.date().format("YYYYMM"))


    ########################################################################################

    world_df = pd.read_csv('../data/Global_Samples_Balanced_n100_m250_s210_attempt1.csv', index_col=[0])
    world_df.head()
    world_df['geometry'] = world_df['geometry'].apply(wkt.loads)
    world_df['samples'] = world_df['samples'].apply(literal_eval)
    world_df['n_samples'] = world_df.apply(lambda row: len(row['samples']), axis=1)
    world_gdf = geopandas.GeoDataFrame(world_df, geometry='geometry')

    country_gdf = world_gdf[(world_gdf['country_code']==country) & (world_gdf['n_samples']>0)]
    country_gdf = country_gdf.reset_index()

    country_polygon_dict = dict()
    for index, row in country_gdf.iterrows():
        country_polygon_dict[country+'_'+str(index)] = (row['geometry'], row['samples'])

    for k, v in country_polygon_dict.items():

        country_segment = k
        print('==================== {} ===================='.format(country_segment))
        area_of_interest_shapely, random_pixels = v

        area_of_interest_ee = ee.Geometry.Polygon(list(area_of_interest_shapely.exterior.coords))


        for month in range(1, 13):
            month_start = datetime.date(year, month, 1)
            month_end = last_day_of_month(datetime.date(year, month, 1))   


            # Create image collection that contains the area of interest
            img_collect = (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                         .filterDate(str(month_start), str(month_end))
                         .filterBounds(area_of_interest_ee))

            if img_collect.size().getInfo() == 0:
                warnings.warn('No valid image.')
                continue
            print("Total number of images in the collection: ", img_collect.size().getInfo())

            img_collect_calc = img_collect.map(calc_YYYYMM)

            unique_month = list(set([item['properties']['YYYYMM'] for item in img_collect_calc.getInfo()['features']]))
            unique_month.sort()
            print(unique_month)

            img_calc_month_dict = dict()
            data_dict = dict()
            for month in unique_month:
                img_calc_month_dict[month] = img_collect_calc.filter(ee.Filter.eq('YYYYMM',month)).first()
                img_calc_month2 = img_calc_month_dict[month].addBands(ee.Image.pixelLonLat())
                data_dict[month] = pd.DataFrame(columns=["lat", "lon", month+'_aet'])
                for p in range(len(random_pixels)):
                    # `reduceRegion` doesn't like ee.Geometry.MultiPoint as geometry
                    data_month_lst = img_calc_month2.reduceRegion(reducer=ee.Reducer.toList(), \
                                                                     geometry=ee.Geometry.Point(random_pixels[p]), maxPixels=1e13, scale=resolution)

                    # For some reason, ee.Array(data_month_lst.get("...")).getInfo()[0] runs really slow
                    pixel_dict = dict()
                    pixel_dict['lat'] = ee.Array(data_month_lst.get("latitude")).getInfo()[0]
                    pixel_dict['lon'] = ee.Array(data_month_lst.get("longitude")).getInfo()[0]
                    try:
                        pixel_dict[month+'_aet'] = ee.Array(data_month_lst.get("aet")).getInfo()[0]
                        pixel_dict[month+'_def'] = ee.Array(data_month_lst.get("def")).getInfo()[0]
                        pixel_dict[month+'_pdsi'] = ee.Array(data_month_lst.get("pdsi")).getInfo()[0]
                        pixel_dict[month+'_pet'] = ee.Array(data_month_lst.get("pet")).getInfo()[0]
                        pixel_dict[month+'_pr'] = ee.Array(data_month_lst.get("pr")).getInfo()[0]
                        pixel_dict[month+'_ro'] = ee.Array(data_month_lst.get("ro")).getInfo()[0]
                        pixel_dict[month+'_soil'] = ee.Array(data_month_lst.get("soil")).getInfo()[0]
                        pixel_dict[month+'_srad'] = ee.Array(data_month_lst.get("srad")).getInfo()[0]
                        pixel_dict[month+'_swe'] = ee.Array(data_month_lst.get("swe")).getInfo()[0]
                        pixel_dict[month+'_tmmn'] = ee.Array(data_month_lst.get("tmmn")).getInfo()[0]
                        pixel_dict[month+'_tmmx'] = ee.Array(data_month_lst.get("tmmx")).getInfo()[0]
                        pixel_dict[month+'_vap'] = ee.Array(data_month_lst.get("vap")).getInfo()[0]
                        pixel_dict[month+'_vpd'] = ee.Array(data_month_lst.get("vpd")).getInfo()[0]
                        pixel_dict[month+'_vs'] = ee.Array(data_month_lst.get("vs")).getInfo()[0]
                    except:
                        warnings.warn('Missing TerraClimate data.')
                        pixel_dict[month+'_aet'] = None
                        pixel_dict[month+'_def'] = None
                        pixel_dict[month+'_pdsi'] = None
                        pixel_dict[month+'_pet'] = None
                        pixel_dict[month+'_pr'] = None
                        pixel_dict[month+'_ro'] = None
                        pixel_dict[month+'_soil'] = None
                        pixel_dict[month+'_srad'] = None
                        pixel_dict[month+'_swe'] = None
                        pixel_dict[month+'_tmmn'] = None
                        pixel_dict[month+'_tmmx'] = None
                        pixel_dict[month+'_vap'] = None
                        pixel_dict[month+'_vpd'] = None
                        pixel_dict[month+'_vs'] = None

                    data_dict[month] = data_dict[month].append(pixel_dict, ignore_index=True)
                data_dict[month].to_csv('../data/GEE/TerraClimate_samples/{}_{}.csv'.format(country_segment, month))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Pull monthly climate data from TerraClimate on sampled lat/lon.')
    parser.add_argument('country', type=str, help='ISO alpha-3 country code')
    parser.add_argument('year', type=int, help='year of TerraClimate data you want to pull')
    parser.add_argument('--resolution', type=int, default=30, help='resolution of TerraClimate data you want to pull')
    args = parser.parse_args()

    start = time.time()
    
    main(args.country, args.year, args.resolution)
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))