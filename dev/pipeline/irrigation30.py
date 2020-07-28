import time
import pandas as pd
#import geopandas
from math import sin, cos, sqrt, atan2, radians
from shapely.geometry import box
import ee
import folium
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
        
class irrigation30():
    
    maxClusters_set = 2
    ndvi_lst = ['ndvi'+str(i).zfill(2) for i in range(1, 13)]
    
    def __init__(self, center_lat=43.771114, center_lon=-116.736866, edge_len=0.005, year=2018, resolution=30, maxSample = 200000):
        '''
        Parameters: 
            center_lat: latitude for the location coordinate
            center_lon: longitude for the location coordinate
            edge_len: edge length for the rectangle given the location coordinates
            year: year the satellite data should pull images for
            resolution: resolution for the image information pull. Default is 30m'''

        # imports

        
        # Trigger the authentication flow.
#         ee.Authenticate()

        # Initialize the library.
        ee.Initialize()
         
        if type(center_lat) == float:
            self.center_lat = center_lat
        else:
            raise ValueError('Please enter float value for latitude')
            exit()
        
        if type(center_lon) == float:
            self.center_lon = center_lon
        else:
            raise ValueError('Please enter float value for longitude')
            exit()
            
        if type(edge_len) == float:
            self.edge_len = edge_len
        else:
            raise ValueError('Please enter float value for edge length')
            exit()
        
        if ((type(year) == int)  and (year > 2015 ) and year <= int(time.strftime("%Y"))):
            self.year = year
        else:
            raise ValueError('Please enter integer value for year > 2015 and less than or equal to current year')
            exit()
         
        if ((type(resolution) == int) and (resolution >=10)):
            self.resolution = resolution
        else:
            raise ValueError('Please enter integer value for resolution greater than or equal to 10')
            exit()
        
        if type(maxSample) == int:
            self.maxSample = maxSample
        else:
            raise ValueError('Please enter integer value for maxSample')
            exit()
            
        self.label = []
        self.avg_ndvi = np.zeros((2, 12))
#         self.std_ndvi = np.zeros((2, 12))
        self.image = ee.Image()
        self.nClusters = 0
            
        self.aoi_ee = self.__create_bounding_box_ee()
        self.dist_lon = self.__calc_distance(self.center_lon-self.edge_len/2, self.center_lat, self.center_lon+self.edge_len/2, self.center_lat)
        self.dist_lat = self.__calc_distance(self.center_lon, self.center_lat-self.edge_len/2, self.center_lon, self.center_lat+self.edge_len/2)
        self.predicted_image = ee.Image()
        print('The selected area is approximately {:.2f} km by {:.2f} km'.format(self.dist_lon, self.dist_lat))
        
        est_total_pixels = round(self.dist_lat*self.dist_lon*(1000**2)/((self.resolution)**2))
#         self.nSample = min(irrigation30.maxSample,est_total_pixels)
#         print('The estimated percentage of pixels used in the model is {:.0%}.'.format(self.nSample/est_total_pixels))
        self.nSample = min(self.maxSample,est_total_pixels)
        pix_percent = self.nSample/est_total_pixels
        print('The estimated percentage of pixels used in the model is {:.0%}.'.format(pix_percent))

        if round(pix_percent*100) < 6:
            raise RuntimeError('The percentange pixel selection is too low for estimating fit predict. Please select a lesser edge length or higher number of sample pixels.')
        if self.edge_len < 0.005:
            raise RuntimeError('Please select an edge length greater than or equal to 0.005 degree.')
        # if year < 2018:
        #     raise RuntimeError('Please select a year greater than or equal to 2018.')
        if self.resolution < 10:
            raise RuntimeError('Please select a resolution greater than or equal to 10.')
        if self.maxSample > 200000:
            raise RuntimeError('The sample range is too high for GEE to handle. Maximum number of pixels that can be processed without Compute error is maxSample = 200000.')

        # hard-code a few things
        # base_asset_directory is where we are going to store output images
        self.base_asset_directory = "users/mbrimmer/w210_irrigated_croplands"
        self.model_projection = "EPSG:4326"
        self.testing_asset_folder = self.base_asset_directory + '/testing/'

    def __create_bounding_box_ee(self):
        '''Creates a rectangle for pulling image information using center coordinates and edge_len'''
        return ee.Geometry.Rectangle([self.center_lon-self.edge_len/2, self.center_lat-self.edge_len/2, self.center_lon+self.edge_len/2, self.center_lat+self.edge_len/2])
    
    def __create_bounding_box_shapely(self):
        '''Returns a box for coordinates to plug in as an image add-on layer'''
        return box(self.center_lon-self.edge_len/2, self.center_lat-self.edge_len/2, self.center_lon+self.edge_len/2, self.center_lat+self.edge_len/2)

    @staticmethod
    def __calc_distance(lon1, lat1, lon2, lat2):
            '''Calculates the distance between 2 coordinates'''
            # Reference: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
            # approximate radius of earth in km
            R = 6373.0
            lon1 = radians(lon1)
            lat1 = radians(lat1)
            lon2 = radians(lon2)
            lat2 = radians(lat2)
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c
            return distance
        
    def __pull_Sentinel2_data(self):
        '''Output monthly Sentinel image dataset for a specified area with NDVI readings for the year 
        merged with GFSAD30 and GFSAD1000 information'''
        band_blue = 'B2' #10m
        band_green = 'B3' #10m
        band_red = "B4"  #10m
        band_nir = 'B8'  #10m   
        
        start_date = str(self.year)+'-1-01'
        end_date = str(self.year)+'-12-31'
        
        # Create image collection that contains the area of interest
        Sentinel_IC = (ee.ImageCollection('COPERNICUS/S2')
                         .filterDate(start_date, end_date)
                         .filterBounds(self.aoi_ee)
                         .select(band_nir, band_red))
        
        # # Get GFSAD30
        GFSAD30_IC = ee.ImageCollection("users/ajsohn/GFSAD30").filterBounds(self.aoi_ee)
        GFSAD30_img = GFSAD30_IC.max().clip(self.aoi_ee)

        def __calc_NDVI(img):
            '''A function to compute Normalized Difference Vegetation Index'''
            ndvi = ee.Image(img.normalizedDifference([band_nir, band_red])).rename(["ndvi"]).copyProperties(img, img.propertyNames())
            composite = img.addBands(ndvi)
            return composite
        
        def __get_by_month_data(img):
            '''Returns an image after merging the ndvi readings and GFSAD30 data per month'''
            months = ee.List.sequence(1,12)
            byMonth = ee.ImageCollection.fromImages(
                months.map(lambda m: img.filter(ee.Filter.calendarRange(m, m, 'month')).median().set('month', m)
                          ).flatten())

            # Take all the satellite bands that have been split into months 
            # as different images in collection (byMonth), and merge into different bands
            def __mergeBands(image, previous):
                '''Returns an image after merging the image with previous image'''
                return ee.Image(previous).addBands(image).copyProperties(image, image.propertyNames())

            merged = byMonth.iterate(__mergeBands, ee.Image())
            return ee.Image(merged).select(['ndvi']+['ndvi_'+str(i) for i in range(1,12)],
                irrigation30.ndvi_lst)
        
        Sentinel_IC = Sentinel_IC.map(__calc_NDVI).select('ndvi')

        # ---------- GET MONTHLY DATA ---------
        # 2 = cropland, 0 - water, 1 = non-cropland, 3 = no-data
        byMonth_img = __get_by_month_data(Sentinel_IC) \
                        .addBands(GFSAD30_img.rename(['gfsad30'])) \
                        .addBands(ee.Image("USGS/GFSAD1000_V1").rename(['gfsad1000'])) \
                        .clip(self.aoi_ee)
    
        # Mask the non-cropland
        # 0 = water, 1 = non-cropland, 2 = cropland, 3 = 'no data'
        cropland = byMonth_img.select('gfsad30').eq(2)
        byMonth_img_masked = byMonth_img.mask(cropland)
#         non_cropland = byMonth_img.select('gfsad30').lt(2) or byMonth_img.select('gfsad30').gt(2)
#         non_cropland = non_cropland.mask(non_cropland)

        return byMonth_img_masked

    def __pull_TerraClimate_data(self, band, label, multiplier=1):       
        '''Output monthly TerraClimate image dataset for a specified area for the year'''
        start_date = str(self.year)+'-1-01'
        end_date = str(self.year)+'-12-31'
        
        # Create image collection that contains the area of interest
        TerraClimate_IC = (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                         .filterDate(start_date, end_date)
                         .filterBounds(self.aoi_ee)
                         .select(band))
        
        def __get_by_month_data(img):
            '''Returns an image after merging the band readings per month'''
            months = ee.List.sequence(1,12)
            byMonth = ee.ImageCollection.fromImages(
                months.map(lambda m: img.filter(ee.Filter.calendarRange(m, m, 'month')).median().set('month', m)
                          ).flatten())

            # Take all the satellite bands that have been split into months 
            # as different images in collection (byMonth), and merge into different bands
            def __mergeBands(image, previous):
                '''Returns an image after merging the image with previous image'''
                return ee.Image(previous).addBands(image).copyProperties(image, image.propertyNames())

            merged = byMonth.iterate(__mergeBands, ee.Image())
            return ee.Image(merged).select([band]+[band+'_'+str(i) for i in range(1,12)],
                [band+str(i).zfill(2) for i in range(1, 13)])

        # ---------- GET MONTHLY DATA ---------
        # 2 = cropland, 0 - water, 1 = non-cropland, 3 = no-data
        byMonth_img = __get_by_month_data(TerraClimate_IC).clip(self.aoi_ee)

        pr_dict = byMonth_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=self.aoi_ee, maxPixels=1e13, scale=self.resolution).getInfo()
        pr_df = pd.DataFrame([pr_dict], columns=[band+str(i).zfill(2) for i in range(1, 13)], index=[label])
        pr_arr = pr_df.to_numpy()*multiplier
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(pr_arr[0], label=label)
        plt.legend()
        
    def plot_precipitation(self):
        '''Plots precepitation from TerraClimate'''
        self.__pull_TerraClimate_data('pr', 'Precipitation')
        
    def plot_temperature_max(self):
        '''Plots max temperature from TerraClimate'''
        self.__pull_TerraClimate_data('tmmx', 'Max Temperature', multiplier=0.1)

    def fit_predict(self):
        '''Builds model using sampled ndvi dataset for training'''

#         self.image = self.__pull_Sentinel2_data()
        try:
            self.image = self.__pull_Sentinel2_data()
#             print("image: ", (self.image).getInfo())
        except:
            raise RuntimeError('GEE will run into issues due to missing images')
            
        training_FC = self.image \
                    .select(irrigation30.ndvi_lst) \
                    .sample(region=self.aoi_ee, scale=self.resolution, numPixels=self.nSample)
        
        # Instantiate the clusterer and train it.
#         clusterer = ee.Clusterer.wekaKMeans(irrigation30.maxClusters_set).train(training_FC, inputProperties=irrigation30.ndvi_lst)
        clusterer = ee.Clusterer.wekaCascadeKMeans(minClusters=2, maxClusters=irrigation30.maxClusters_set).train(training_FC, inputProperties=irrigation30.ndvi_lst)
        # wekaXMeans outputs the same number of clusters but different mixes when maxClusters is set differently
#         clusterer = ee.Clusterer.wekaXMeans(minClusters=2, maxClusters=irrigation30.maxClusters_set).train(training_FC, inputProperties=irrigation30.ndvi_lst)

        # Cluster the input using the trained clusterer.
        cluster_result = self.image.cluster(clusterer)
        
        cluster_output = dict()
        for i in range(0, irrigation30.maxClusters_set):
            print('Averaging NDVIs for Cluster {}...'.format(i))
            cluster_output[i] = self.image.select(irrigation30.ndvi_lst).mask(cluster_result.select('cluster').eq(i)).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.aoi_ee, maxPixels=1e13, scale=30).getInfo()
            if cluster_output[i]['ndvi01']==None:
                self.nClusters = i
                del cluster_output[i]
                break
            elif i == irrigation30.maxClusters_set-1:
                self.nClusters = irrigation30.maxClusters_set
        
        # Reference: https://stackoverflow.com/questions/45194934/eval-fails-in-list-comprehension
        globs = globals()
        locs = locals()
        cluster_df = pd.DataFrame([eval('cluster_output[{}]'.format(i), globs, locs) for i in range(0,self.nClusters)], columns=irrigation30.ndvi_lst, index=['Cluster_'+str(i) for i in range(0,self.nClusters)])

        self.avg_ndvi = cluster_df.to_numpy()
        cluster_mean = self.avg_ndvi.mean(axis=1)
        if self.nClusters == 2:
            if cluster_mean[0] < cluster_mean[1]:
                self.image = self.image.addBands(ee.Image(cluster_result.select('cluster')).rename('prediction'))
                self.label = ["Rainfed", "Irrigated"]
            else:
                self.image = self.image.addBands(ee.Image(cluster_result.expression('1-c',{'c':cluster_result.select('cluster')})).rename('prediction'))
                self.label = ["Irrigated", "Rainfed"]
        else:
            self.image = self.image.addBands(ee.Image(cluster_result.select('cluster')).rename('prediction'))
            self.label = ['Cluster_'+str(i) for i in range(0,self.nClusters)]

        # Now create single image that will be the prediction for the area
        # first need to create image with a single band for the prediction and the GFSAD information
        GFSAD30_IC = ee.ImageCollection("users/ajsohn/GFSAD30").filterBounds(self.aoi_ee)
        GFSAD30_img = GFSAD30_IC.max().clip(self.aoi_ee)

        filterTimeEq = ee.Filter.equals(leftField= '1',rightField= '1')
        res_w_gfsad30_temp = ee.Join.inner().apply(ee.ImageCollection([cluster_result]), GFSAD30_IC, filterTimeEq);
        res_w_gfsad30 = res_w_gfsad30_temp.map( lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
        res_w_gfsad30_img = ee.ImageCollection(res_w_gfsad30).median()

        # Create single band image with results and gfsad
        if self.label[0] == "Rainfed":
            predicted_image = res_w_gfsad30_img.expression(
            "(b('b1') != 2) ? 0 " +
                ": (b('cluster') == 0) ? 0 : 1"
                ).rename('class')
        else:
            predicted_image = res_w_gfsad30_img.expression(
            "(b('b1') != 2) ? 0 " +
                ": (b('cluster') == 1) ? 0 : 1"
                ).rename('class')


        
#         print('Calculating standard deviation of NDVIs for Cluster 0...')
#         cluster_0_dict = self.image.select(irrigation30.ndvi_lst).mask(cluster_result.select('cluster').eq(0)).reduceRegion(reducer=ee.Reducer.stdDev(), geometry=self.aoi_ee, maxPixels=1e13, scale=self.resolution).getInfo()
#         print('Calculating standard deviation of NDVIs for Cluster 1...')
#         cluster_1_dict = self.image.select(irrigation30.ndvi_lst).mask(cluster_result.select('cluster').eq(1)).reduceRegion(reducer=ee.Reducer.stdDev(), geometry=self.aoi_ee, maxPixels=1e13, scale=self.resolution).getInfo()
#         cluster_df = pd.DataFrame([cluster_0_dict, cluster_1_dict], columns=irrigation30.ndvi_lst, index=['Cluster_0', 'Cluster_1'])
#         self.std_ndvi = cluster_df.to_numpy()
        
        print('Model complete')
        
    def plot_map(self):
        '''Plot folium map using GEE api - the map includes are of interest box and associated ndvi readings'''
        def add_ee_layer(self, ee_object, vis_params, show, name):
            '''Checks if image object classifies as ImageCollection, FeatureCollection, Geometry or single Image
            and adds to folium map accordingly'''
            try:    
                if isinstance(ee_object, ee.image.Image):    
                    map_id_dict = ee.Image(ee_object).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles = map_id_dict['tile_fetcher'].url_format,
                        attr = 'Google Earth Engine',
                        name = name,
                        overlay = True,
                        control = True,
                        show = show
                        ).add_to(self)
                elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
                    ee_object_new = ee_object.median()
                    map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                        tiles = map_id_dict['tile_fetcher'].url_format,
                        attr = 'Google Earth Engine',
                        name = name,
                        overlay = True,
                        control = True,
                        show = show
                        ).add_to(self)
                elif isinstance(ee_object, ee.geometry.Geometry):    
                    folium.GeoJson(
                            data = ee_object.getInfo(),
                            name = name,
                            overlay = True,
                            control = True
                        ).add_to(self)
                elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
                    ee_object_new = ee.Image().paint(ee_object, 0, 2)
                    map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
                    folium.raster_layers.TileLayer(
                            tiles = map_id_dict['tile_fetcher'].url_format,
                            attr = 'Google Earth Engine',
                            name = name,
                            overlay = True,
                            control = True,
                            show = show
                        ).add_to(self)

            except:
                print("Could not display {}".format(name))

        # Add EE drawing method to folium.
        folium.Map.add_ee_layer = add_ee_layer

        myMap = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=8)
        aoi_shapely = self.__create_bounding_box_shapely()
        folium.GeoJson(aoi_shapely, name="Area of Interest").add_to(myMap)
        visParams = {'min':0, 'max':1, 'palette': ['yellow', 'green']}
        myMap.add_ee_layer(self.image.select('prediction'), visParams, show=True, name='Prediction')
        #     0: Non-croplands (black)
        #     1: Croplands: irrigation major (green)
        #     2: Croplands: irrigation minor (lighter green)
        #     3: Croplands: rainfed (yellow)
        #     4: Croplands: rainfed, minor fragments (yellow orange)
        #     5: Croplands: rainfed, rainfed, very minor fragments (orange)
        visParams = {'min':0, 'max':5, 'palette':['black', 'green', 'a9e1a9', 'yellow', 'ffdb00', '#ffa500']}
        myMap.add_ee_layer(self.image.select('gfsad1000'), visParams, show=False, name='GFSAD1000')
        visParams = {'min':0, 'max':1, 'palette': ['red', 'yellow', 'green']}
        for i in range(1, 13):
            temp_band = 'ndvi'+str(i).zfill(2) 
            myMap.add_ee_layer(self.image.select(temp_band), visParams, show=False, name=temp_band)
        myMap.add_child(folium.LayerControl())

        return myMap
    
    def plot_avg_ndvi(self):
        '''Plotting for ndvi readings'''
        fig, ax = plt.subplots(figsize=(12, 6))
        if self.nClusters == 2:
            plt.plot(self.avg_ndvi[0], label=self.label[0])
            plt.plot(self.avg_ndvi[1], label=self.label[1])
        else:
            for i in range(0, self.nClusters):
                plt.plot(self.avg_ndvi[i], label='Cluster_'+str(i))
        plt.legend()

    def __smooth(self, y_raw):
        '''Smoothing points for plotting by adding points to yearly curve at the beginning and end of curve'''
        y = np.concatenate((y_raw, y_raw, y_raw))
        x = np.linspace(0, 35, num=36, endpoint=True)
        xnew = np.linspace(0, 35, num=141, endpoint=True)
        f_interp = interp1d(x, y, kind='cubic')
        y_interp = f_interp(xnew)
        y_loess = savgol_filter(y_interp, 7, 1)
        peak_index, peak_value = find_peaks(y_loess, height=0)
        # Intermediate graph
        # plt.plot(x, y, 'o', xnew, y_loess, '--')
        # plt.plot(xnew[peak_index], peak_value['peak_heights'], "x")
        # plt.legend(['data', 'smooth'], loc='best')
        # plt.show()
        # Final graph
        final_peak_index = [i - 12 for i in xnew[peak_index] if i >= 12 and i < 24]
        print(final_peak_index)
        final_peak_value = [j for i, j in zip(xnew[peak_index], peak_value['peak_heights']) if i >= 12 and i < 24]
        final_x = [i - 12 for i in xnew if i >= 12 and i < 24]
        final_y_loess = [j for i, j in zip(xnew, y_loess) if i >= 12 and i < 24]
        return final_x, final_y_loess, final_peak_index, final_peak_value
        
    def predict_crop_season(self):
        '''Plots smoothed ndvi for irrigated vs non-irrigated'''
        if self.label[0] == 'Irrigated':
            y_raw = self.avg_ndvi[0] - self.avg_ndvi[1]
        else:
            y_raw = self.avg_ndvi[1] - self.avg_ndvi[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        y_raw = self.avg_ndvi[0]
        final_x, final_y_loess, final_peak_index, final_peak_value = self.__smooth(y_raw)
        plt.plot(range(0,12), y_raw, 'o', final_x, final_y_loess, '--')
        plt.plot(final_peak_index, final_peak_value, "x")
        y_raw = self.avg_ndvi[1]
        final_x, final_y_loess, final_peak_index, final_peak_value = self.__smooth(y_raw)
        plt.plot(range(0,12), y_raw, 'o', final_x, final_y_loess, '--')
        plt.plot(final_peak_index, final_peak_value, "x")
#         plt.legend(['Diff in NDVI', 'Smooth'], loc='best')


    def write_image_asset(self, image_asset_id, wait=False):
        '''Writes predicted image out as an image to Google Earth Engine as an asset'''
        image_asset_id = self.base_asset_directory + '/' +  image_asset_id

        task = ee.batch.Export.image.toAsset(
            crs=self.model_projection,
            region=self.aoi_ee,
            image=self.predicted_image,
            scale=30,
            assetId=image_asset_id,
            maxPixels=1e13
        )
        task.start()


    def write_image_google_drive(self, filename):
        '''Writes predicted image out as an image to Google Drive as a TIF file'''
        task = ee.batch.Export.image.toDrive(
            crs=self.model_projection,
            region=self.aoi_ee,
            image=self.predicted_image,
            scale=30,
            description=filename,
            maxPixels=1e13
        )
        print("Writing To Google Drive filename= ", filename)
        task.start()

