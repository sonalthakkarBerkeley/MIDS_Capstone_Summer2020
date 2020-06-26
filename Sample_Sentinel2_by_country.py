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
    
    # Reference: https://www.satimagingcorp.com/satellite-sensors/other-satellite-sensors/sentinel-2a/
    band_blue = 'B2' #10m
    band_green = 'B3' #10m
    band_red = "B4"  #10m
    band_nir = 'B8'  #10m
    
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
                   
    def add_landcover(img):
        landcover = ee.Image("USGS/GFSAD1000_V1")
        composite = img.addBands(landcover)
        return composite
    
    def calc_YYYYMM(img):
        return img.set('YYYYMM', img.date().format("YYYYMM"))
    
    
    ########################################################################################
    
    world_df = pd.read_csv('../data/Global_samples_Balanced_100.csv', index_col=[0])
    world_df.head()
    world_df['geometry'] = world_df['geometry'].apply(wkt.loads)
    world_df['samples'] = world_df['samples'].apply(literal_eval)
    world_gdf = geopandas.GeoDataFrame(world_df, geometry='geometry')
    
    country_gdf = world_gdf[(world_gdf['country_code']==country) & (world_gdf['area']>2)]
    country_gdf = country_gdf.reset_index()
    
    country_polygon_dict = dict()
    for index, row in country_gdf.iterrows():
        country_polygon_dict[country+'_'+str(index)] = (row['geometry'], row['samples'])
    
    for k, v in country_polygon_dict.items():
    
        country_segment = k
        area_of_interest_shapely, random_pixels = v
        
        area_of_interest_ee = ee.Geometry.Polygon(list(area_of_interest_shapely.exterior.coords))


        for month in range(1, 13):
            month_start = datetime.date(year, month, 1)
            month_end = last_day_of_month(datetime.date(year, month, 1))   

    
            # Create image collection that contains the area of interest
            img_collect = (ee.ImageCollection('COPERNICUS/S2')
                         .filterDate(str(month_start), str(month_end))
                         .filterBounds(area_of_interest_ee)
                            # Remove image that's too small (likely to be partial image)
                            # Size of a full image: 1,276,131,371; size of a partial image: 276,598,191
#                          .filter(ee.Filter.gt('system:asset_size', 800000000))
                         .filterMetadata("CLOUDY_PIXEL_PERCENTAGE","less_than",50)
                          )


            assert (img_collect.size().getInfo()>0), "No valid image"
            print("Total number of images in the collection: ", img_collect.size().getInfo())

            # Extract tile information from each image
            # Note: tiles can overlap a little bit
            unique_tiles = set([item['properties']['MGRS_TILE'] for item in img_collect.getInfo()['features']])
            if len(unique_tiles) > 1:
                print('Number of tiles selected: ', len(unique_tiles))
#             if img_collect_no_partial.size().getInfo() < img_collect.size().getInfo():
#                 warnings.warn('There are partial images in the collection. Proceed with caution.')
#                 print('Number of partial images: ', img_collect.size().getInfo()-img_collect_no_partial.size().getInfo())


            img_collect_calc = img_collect.map(calc_YYYYMM).map(calc_NDVI).map(calc_SAVI).map(calc_EVI).map(add_landcover)

            unique_month = list(set([item['properties']['YYYYMM'] for item in img_collect_calc.getInfo()['features']]))
            unique_month.sort()
            print(unique_month)
    
#     1. min_lat, min_lon, max_lat, max_lon for the country
#             min_lon, min_lat, max_lon, max_lat = area_of_interest_shapely.bounds
#     2. from left to right, top to bottom, define 0.1 degree by 0.1 degree overlapping rectangles
#     3. only keep the rectangles that's within the country's boundary (possbily using spatial join)
#     4. for rectangle in a list of recntagles:
#         .reduceRegion(geometry=rectangle)
#         save to csv
    
    
            img_calc_month_dict = dict()
            data_dict = dict()
            for month in unique_month:
                img_calc_month_dict[month] = img_collect_calc.filter(ee.Filter.eq('YYYYMM',month)).median()
                img_calc_month2 = img_calc_month_dict[month].addBands(ee.Image.pixelLonLat())
                # EEException: Output of image computation is too large (20 bands for 851968 pixels = 126.8 MiB > 80.0 MiB).
                #     If this is a reduction, try specifying a larger 'tileScale' parameter.
                # EEException: ReduceRegion.AggregationContainer: Valid tileScales are 1 to 16.
                data_dict[month] = pd.DataFrame(columns=["lat", "lon", 'landcover', month+'_NDVI', month+'_SAVI', month+'_EVI'])
                for p in range(len(random_pixels)):
                    # `reduceRegion` doesn't like ee.Geometry.MultiPoint as geometry
                    data_month_lst = img_calc_month2.reduceRegion(reducer=ee.Reducer.toList(), \
                                                                     geometry=ee.Geometry.Point(random_pixels[p]), maxPixels=1e13, scale=resolution)

                    # For some reason, ee.Array(data_month_lst.get("...")).getInfo()[0] runs really slow
                    pixel_dict = dict()
                    pixel_dict['lat'] = ee.Array(data_month_lst.get("latitude")).getInfo()[0]
                    pixel_dict['lon'] = ee.Array(data_month_lst.get("longitude")).getInfo()[0]
                    try:
                        pixel_dict['landcover'] = ee.Array(data_month_lst.get("landcover")).getInfo()[0]
                        pixel_dict[month+'_NDVI'] = ee.Array(data_month_lst.get("ndvi")).getInfo()[0]
                        pixel_dict[month+'_SAVI'] = ee.Array(data_month_lst.get("savi")).getInfo()[0]
                        pixel_dict[month+'_EVI'] = ee.Array(data_month_lst.get("evi")).getInfo()[0]
                    except:
                        warnings.warn('Missing satellite data.')
                        pixel_dict['landcover'] = None
                        pixel_dict[month+'_NDVI'] = None
                        pixel_dict[month+'_SAVI'] = None
                        pixel_dict[month+'_EVI'] = None
                    data_dict[month] = data_dict[month].append(pixel_dict, ignore_index=True)
                data_dict[month].to_csv('../data/GEE/Sentinel2_samples/{}_{}.csv'.format(country_segment, month))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Pull vegetation index from Sentinel-2 on sampled lat/lon.')
    parser.add_argument('country', type=str, help='ISO alpha-3 country code')
    parser.add_argument('year', type=int, help='year of Sentinel-2 data you want to pull')
    parser.add_argument('--resolution', type=int, help='resolution of Sentinel-2 dat you want to pull')
    args = parser.parse_args()

    start = time.time()
    
    main(args.country, args.year, args.resolution)
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))