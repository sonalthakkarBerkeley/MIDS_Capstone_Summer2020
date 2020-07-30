# import packages needed
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
 
    # Trigger the authentication flow.
    #ee.Authenticate()

    # Set the max number of samples used in the clustering
    maxSample = 100000
    # Technically, resolution can be a parameter in __init___
    #     But we did not fully test resolutions different from 30 m.
    resolution = 30
    # Reference: https://hess.copernicus.org/articles/19/4441/2015/hessd-12-1329-2015.pdf
    # "If NDVI at peak is less than 0.40, the peak is not counted as cultivation." 
    #     The article uses 10-day composite NDVI while we use montly NDVI.
    #     To account for averaging effect, our threshold is slightly lower than 0.4.
    crop_ndvi_threashold = 0.3
    # Estimated based on http://www.fao.org/3/s2022e/s2022e07.htm#TopOfPage
    water_need_threshold = 100
    # Rename ndvi bands to the following
    ndvi_lst = ['ndvi'+str(i).zfill(2) for i in range(1, 13)]
    # Give descriptive name for the month
    month_lst = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # List of colors used to plot each cluster
    cluster_color = ['red', 'blue', 'orange', 'yellow', 'darkgreen', 'lightgreen', 'lightblue', 'purple', 'pink', 'lightgray']
    
    
    def __init__(self, center_lat=43.771114, center_lon=-116.736866, edge_len=0.005, year=2018, maxClusters_set=2):
        '''
        Parameters: 
            center_lat: latitude for the location coordinate
            center_lon: longitude for the location coordinate
            edge_len: edge length for the rectangle given the location coordinates
            year: year the satellite data should pull images for
            maxClusters_set: should be for range 2-10'''


        # Initialize the library.
        ee.Initialize()
         
        # error handle parameter issues
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
            
        if (type(edge_len) == float and (edge_len <=0.5 and edge_len > 0.005)):
            self.edge_len = edge_len
        else:
            raise ValueError('Please enter float value for edge length between 0.5 and 0.005')
            exit()
            
        # (range is 2017 to year prior)
        if ((type(year) == int)  and (year >= 2017 and year <= int(time.strftime("%Y")) - 1)):
            self.year = year
        else:
            raise ValueError('Please enter integer value for year > 2017 and less than current year')
            exit()
        
        # n_clusters (2-10)
        if ((type(maxClusters_set) == int) and (maxClusters_set >=2 and maxClusters_set <= 10)):
            self.maxClusters_set = maxClusters_set
        else:
            raise ValueError('Please enter integer value for resolution greater than or equal to 10')
            exit()


        # initialize remaining variables
        self.label = []
        self.comment = dict()
        self.avg_ndvi = np.zeros((2, 12))
        self.temperature_max = []
        self.temperature_min = []
        self.temperature_avg = []
        self.precipitation = []
        self.image = ee.Image()

        self.nClusters = 0
        self.simple_label = []
        self.simple_image = ee.Image()

        # Create the bounding box using GEE API
        self.aoi_ee = self.__create_bounding_box_ee()
        # Estimate the area of interest
        self.dist_lon = self.__calc_distance(self.center_lon-self.edge_len/2, self.center_lat, self.center_lon+self.edge_len/2, self.center_lat)
        self.dist_lat = self.__calc_distance(self.center_lon, self.center_lat-self.edge_len/2, self.center_lon, self.center_lat+self.edge_len/2)
        print('The selected area is approximately {:.2f} km by {:.2f} km'.format(self.dist_lon, self.dist_lat))
        
        # Estimate the amount of pixels used in the clustering algorithm
        est_total_pixels = round(self.dist_lat*self.dist_lon*(1000**2)/((irrigation30.resolution)**2))
        self.nSample = min(irrigation30.maxSample,est_total_pixels)
#         print('The estimated percentage of pixels used in the model is {:.0%}.'.format(self.nSample/est_total_pixels))


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
        
        # Get GFSAD30 image and clip to the area of interest
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
        
        # Apply the calculation of NDVI
        Sentinel_IC = Sentinel_IC.map(__calc_NDVI).select('ndvi')

        # ---------- GET MONTHLY DATA ---------
        # Get Sentinel-2 monthly data
        # 0 = water, 1 = non-cropland, 2 = cropland, 3 = 'no data'
        byMonth_img = __get_by_month_data(Sentinel_IC) \
                        .addBands(GFSAD30_img.rename(['gfsad30'])) \
                        .addBands(ee.Image("USGS/GFSAD1000_V1").rename(['gfsad1000'])) \
                        .clip(self.aoi_ee)
    
        # Mask the cropland
        cropland = byMonth_img.select('gfsad30').eq(2)
        byMonth_img_masked = byMonth_img.mask(cropland)

        return byMonth_img_masked
    

    def __pull_TerraClimate_data(self, band, multiplier=1):       
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

        # Get TerraClimate monthly data
        byMonth_img = __get_by_month_data(TerraClimate_IC).clip(self.aoi_ee)
        
        # Calculate the average value by month
        climate_dict = byMonth_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=self.aoi_ee, maxPixels=1e13, scale=irrigation30.resolution).getInfo()
        climate_df = pd.DataFrame([climate_dict], columns=[band+str(i).zfill(2) for i in range(1, 13)], index=[0])
        climate_arr = climate_df.to_numpy()*multiplier
        
        return climate_arr
    
    def __identify_peak(self, y_raw):
        '''Returns peak values and the month for peaking'''
        # Peaks cannot be identified if it's the first or last number in a series
        # To resolve this issue, we copy the series three times
        y = np.concatenate((y_raw, y_raw, y_raw))
        x = np.linspace(0, 35, num=36, endpoint=True)
        peak_index_raw, peak_value_raw = find_peaks(y, height=irrigation30.crop_ndvi_threashold)
        # Sometimes there are multiple peaks in a single crop season
        # 
        index_diff = np.diff(peak_index_raw)
        peak_grp = [0]
        counter = 0
        for i in index_diff:
            if i == 2:
                peak_grp.append(counter)
            else:
                counter+=1
                peak_grp.append(counter)
        peak_grp_series = pd.Series(peak_grp, name='peak_grp')
        peak_index_series = pd.Series(peak_index_raw, name='peak_index')
        peak_value_series = pd.Series(peak_value_raw['peak_heights'], name='peak_value')
        peak_grp_df = pd.concat([peak_grp_series, peak_index_series, peak_value_series], axis=1)
        peak_grp_agg_df = peak_grp_df.groupby('peak_grp').agg({'peak_index':np.mean, 'peak_value':np.max})
        peak_index = peak_grp_agg_df['peak_index'].to_numpy()
        peak_value = peak_grp_agg_df['peak_value'].to_numpy()
        
        peak_lst = [(int(i-12), irrigation30.month_lst[int(i-12)], j) for i, j in zip(peak_index, peak_value) if i >= 12 and i < 24]
        final_peak_index = [i[0] for i in peak_lst]
        final_peak_month = [i[1] for i in peak_lst]
        final_peak_value = [i[2] for i in peak_lst]
        return final_peak_index, final_peak_month, final_peak_value
        
    def __identify_label(self, cluster_result):
        '''Plugs in labels for the irrigated and rainfed areas'''
        def __identify_surrounding_month(value, diff):
            '''For the peaked month returns surrounding month data'''
            new_value = value + diff
            if new_value < 0:
                new_value += 12
            elif new_value >= 12:
                new_value -= 12
            return int(new_value)
        def __calc_effective_precipitation(P):
            '''Calculates and prints irrigation labels based on effective precipitation and temperatures'''
            # Reference: 
            # Pe = 0.8 P - 25 if P > 75 mm/month
            # Pe = 0.6 P - 10 if P < 75 mm/month
            if P >= 75:
                Pe = 0.8*P-25
            else:
                Pe = max(0.6*P-10, 0)
            return Pe
        
        self.label = []
        for i in range(self.nClusters):
            final_peak_index, final_peak_month, final_peak_value = self.__identify_peak(self.avg_ndvi[i])
            if len(final_peak_index)==0:
                self.label.append('Cluster {}: Rainfed'.format(i))
                self.comment[i] = 'rainfed'
            else:
                temp_label = []
                temp_comment = '{}-crop cycle annually | '.format(len(final_peak_index))
                if len(self.precipitation) == 0:
                    self.precipitation = self.__pull_TerraClimate_data('pr')[0]
                if len(self.temperature_max) == 0:
                    self.temperature_max = self.__pull_TerraClimate_data('tmmx', multiplier=0.1)[0]
                    self.temperature_min = self.__pull_TerraClimate_data('tmmn', multiplier=0.1)[0]
                self.temperature_avg = np.mean([self.temperature_max, self.temperature_min], axis=0)
                for p in range(len(final_peak_index)):
                    p_index = final_peak_index[p]
                    # Calcuate the precipiration the month before the peak and at the peak
                    # Depending on whether it's Fresh harvested crop or Dry harvested crop, the water need after the mid-season is different
                    # Reference: http://www.fao.org/3/s2022e/s2022e02.htm#TopOfPage
                    p_lst = [__identify_surrounding_month(p_index, -1), p_index]
                    pr_mean = self.precipitation[p_lst].mean()
                    # Lower temperature reduces water need
                    # Reference: http://www.fao.org/3/s2022e/s2022e02.htm#TopOfPage
                    if self.temperature_avg[p_lst].mean() < 15:
                        temperature_adj = 0.7
                    else:
                        temperature_adj = 1
                    if pr_mean >= irrigation30.water_need_threshold * temperature_adj:
                        temp_label.append('Rainfed')
                        temp_comment = temp_comment + 'rainfed around {}; '.format(final_peak_month[p])
                    else:
                        temp_label.append('Irrigated')
                        temp_comment = temp_comment + 'irrigated around {}; '.format(final_peak_month[p])
                self.label.append('Cluster {}: '.format(i)+'+'.join(temp_label))
                self.comment[i] = temp_comment
        self.simple_label = ['Irrigated' if 'Irrigated' in i else 'Rainfed' for i in self.label]
        self.image = self.image.addBands(ee.Image(cluster_result.select('cluster')).rename('prediction'))  
        
    def plot_precipitation(self):
        '''Plots precepitation from TerraClimate'''
        if len(self.precipitation) == 0:
            self.precipitation = self.__pull_TerraClimate_data('pr')[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(irrigation30.month_lst, self.precipitation, label='Precipitation')
        plt.legend()
        
    def plot_temperature_max_min(self):
        '''Plots max and min temperature from TerraClimate'''
        self.temperature_max = self.__pull_TerraClimate_data('tmmx', multiplier=0.1)[0]
        self.temperature_min = self.__pull_TerraClimate_data('tmmn', multiplier=0.1)[0]
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(irrigation30.month_lst, self.temperature_max, label='Max Temperature')
        plt.plot(irrigation30.month_lst, self.temperature_min, label='Min Temperature')
        plt.legend()

    def fit_predict(self):
        '''Builds model using startified datapoints from sampled ndvi dataset for training'''
        
        try:
            self.image = self.__pull_Sentinel2_data()
        except:
            raise RuntimeError('GEE will run into issues due to missing images')
        
        training_FC = self.image.cast({'gfsad30':"int8"},['gfsad30', 'gfsad1000']+irrigation30.ndvi_lst)\
                        .stratifiedSample(region=self.aoi_ee, classBand = 'gfsad30', numPoints = self.nSample,
                        classValues = [0, 1, 3], 
                        classPoints = [0, 0, 0],
                        scale=irrigation30.resolution)\
                        .select(irrigation30.ndvi_lst)
        
        # Instantiate the clusterer and train it.
        clusterer = ee.Clusterer.wekaKMeans(self.maxClusters_set).train(training_FC, inputProperties=irrigation30.ndvi_lst)
        # wekaCascadeKMeans takes much longer to run when maxClusters is greater than minClusters
#         clusterer = ee.Clusterer.wekaCascadeKMeans(minClusters=2, maxClusters=self.maxClusters_set).train(training_FC, inputProperties=irrigation30.ndvi_lst)
#         clusterer = ee.Clusterer.wekaXMeans(minClusters=2, maxClusters=self.maxClusters_set).train(training_FC, inputProperties=irrigation30.ndvi_lst)

        # Cluster the input using the trained clusterer.
        cluster_result = self.image.cluster(clusterer)
        
        print('Model building...')
        cluster_output = dict()
        for i in range(0, self.maxClusters_set):
            cluster_output[i] = self.image.select(irrigation30.ndvi_lst).mask(cluster_result.select('cluster').eq(i)).reduceRegion(reducer=ee.Reducer.mean(), geometry=self.aoi_ee, maxPixels=1e13, scale=30).getInfo()
            if cluster_output[i]['ndvi01']==None:
                self.nClusters = i
                del cluster_output[i]
                break
            elif i == self.maxClusters_set-1:
                self.nClusters = self.maxClusters_set
        
        # Reference: https://stackoverflow.com/questions/45194934/eval-fails-in-list-comprehension
        globs = globals()
        locs = locals()
        cluster_df = pd.DataFrame([eval('cluster_output[{}]'.format(i), globs, locs) for i in range(0,self.nClusters)], columns=irrigation30.ndvi_lst, index=['Cluster_'+str(i) for i in range(0,self.nClusters)])

        self.avg_ndvi = cluster_df.to_numpy()

        self.__identify_label(cluster_result)

        # Binary (simple image) is useful for testing / evaluation purposes
        # print("Print simple_label: ", self.simple_label)
        
        gee_label_irr = ee.List([0] + [1*(self.simple_label[i] == "Irrigated") for i in range(len(self.simple_label))] + [0 for i in range(10-self.nClusters)])

        # including -1 to be my 'not cropland below'
        # Hard-coding 10 as top max_clusters
        cluster_nums_py = [str(i) for i in range(-1,10)]

        cluster_nbrs = ee.List(cluster_nums_py)
        gee_label_dict = ee.Dictionary.fromLists(cluster_nbrs, gee_label_irr)
    
        temp_image = self.image.expression(
            "(b('gfsad30') == 2) ? (b('prediction')) : -1 ").rename('class').cast({'class':'int'}) 

        self.simple_image = temp_image \
            .where(temp_image.eq(-1), ee.Number(0) ) \
            .where(temp_image.eq(0), ee.Number(gee_label_dict.get('0'))) \
            .where(temp_image.eq(1), ee.Number(gee_label_dict.get('1'))) \
            .where(temp_image.eq(2), ee.Number(gee_label_dict.get('2'))) \
            .where(temp_image.eq(3), ee.Number(gee_label_dict.get('3'))) \
            .where(temp_image.eq(4), ee.Number(gee_label_dict.get('4'))) \
            .where(temp_image.eq(5), ee.Number(gee_label_dict.get('5'))) \
            .where(temp_image.eq(6), ee.Number(gee_label_dict.get('6'))) \
            .where(temp_image.eq(7), ee.Number(gee_label_dict.get('7'))) \
            .where(temp_image.eq(8), ee.Number(gee_label_dict.get('8'))) \
            .where(temp_image.eq(9), ee.Number(gee_label_dict.get('9')))

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
            month_label = irrigation30.month_lst[i-1]
            myMap.add_ee_layer(self.image.select(temp_band), visParams, show=False, name='NDVI '+month_label)
        myMap.add_child(folium.LayerControl())
        folium.Marker([self.center_lat, self.center_lon], tooltip='center').add_to(myMap)
        
        print('============ Prediction Layer Legend ============')
        # print the comments for each cluster
        for i in range(self.nClusters):
            print('Cluster {} ({}): {}'.format(i, irrigation30.cluster_color[i], self.comment[i]))
        print('============ GFSAD1000 Layer Legend ============')
        print('Croplands: irrigation major (green)')
        print('Croplands: irrigation minor (lighter green)')
        print('Croplands: rainfed (yellow)')
        print('Croplands: rainfed, minor fragments (yellow orange)')
        print('Croplands: rainfed, rainfed, very minor fragments (orange)')
        print('================================================')
        return myMap
    
    def plot_avg_ndvi(self):
        '''Plotting for ndvi readings'''
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(0, self.nClusters):
            plt.plot(irrigation30.month_lst, self.avg_ndvi[i], label=self.label[i], color=irrigation30.cluster_color[i])
        plt.legend()



    def write_image_asset(self, image_asset_id, write_binary_version = False):
        '''Writes predicted image out as an image to Google Earth Engine as an asset'''
        image_asset_id = self.base_asset_directory + '/' +  image_asset_id

        if write_binary_version == False:
            task = ee.batch.Export.image.toAsset(
                crs=self.model_projection,
                region=self.aoi_ee,
                image=self.image,
                scale=resolution,
                assetId=image_asset_id,
                maxPixels=1e13
            )
            task.start()
        else:
            task = ee.batch.Export.image.toAsset(
                crs=self.model_projection,
                region=self.aoi_ee,
                image=self.simple_image,
                scale=resolution,
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
            scale=resolution,
            description=filename,
            maxPixels=1e13
        )
        print("Writing To Google Drive filename= ", filename)
        task.start()