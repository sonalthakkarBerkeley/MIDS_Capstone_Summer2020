import irrigation30
import math
import ee

base_asset_directory = "users/mbrimmer/w210_irrigated_croplands"

def create_overlapping_images(irr, case_num):
    '''
    This function will create the images in GEE
    '''

    nSegments = 3 # make this an odd number > 1

    base_edge = irr.edge_len
    base_lat = irr.center_lat
    base_lon = irr.center_lon

    step_size = base_edge / nSegments

    latitude_vals = [base_lat + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]
    longitude_vals = [base_lon + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]

    # Build Base Table -- will add to this through iterations
    irr.fit_predict()
    irr.write_image_asset('testing/overlap_test_image_base_case_'+str(case_num), write_binary_version=True)

    i = 0
    for lat in latitude_vals:
        for lon in longitude_vals:
            i = i+1
            # Don't rerun base case because we built that to begin with (above)
            if lat == base_lat and lon == base_lon:
                continue
            else:
                print("Creating image i=",i)
                
                irr_overlap = irrigation30.irrigation30(lat, lon, edge_len=base_edge)
                irr_overlap.fit_predict()
                irr_overlap.write_image_asset('testing/overlap_test_image_case_' + str(case_num) + '_' + str(i), write_binary_version = True)


def evaluate_overlapping_images(nSegments, base_irr, case_num):
    base_img = ee.Image(base_asset_directory + "/testing/"+ "overlap_test_image_base_case_" + str(case_num))

    # if there are 5 segments, there are 24 (25-1 that is the base) images
    img_list = ["overlap_test_image_case_" + str(case_num) + '_' +str(i) 
        for i in range(1,nSegments*nSegments+1) if i != math.floor(nSegments * nSegments/2)+1]

    shifted_images = []
    for img in img_list:
        shifted_images.append(ee.Image(base_asset_directory + "/testing/" + img))

    shifted_images.append(base_img)
    IC = ee.ImageCollection(shifted_images)

    count_img = IC.reduce(ee.Reducer.count())
    sum_img = IC.reduce(ee.Reducer.sum())
    
    # Now merge these together
    innerJoin = ee.Join.inner()
    filterTimeEq = ee.Filter.equals(leftField= '1', rightField= '1')

    innerJoined = innerJoin.apply(ee.ImageCollection([count_img]), ee.ImageCollection([sum_img]), filterTimeEq)
    joined = innerJoined.map( lambda feature: ee.Image.cat(feature.get('primary'), feature.get('secondary')))
    joined_img = ee.ImageCollection(joined).max().cast({'class_sum': 'double', 'class_count': 'double'})

    
    # now create percent similar
    percent_similar = joined_img.expression(
        "(b('class_sum') / b('class_count') > 1 - b('class_sum') / b('class_count') ) ? " +
            "b('class_sum') / b('class_count') : 1- b('class_sum') / b('class_count')"
            ).rename('percent_same')

    # Now reduce the image to get mean
    # task = ee.batch.Export.image.toAsset(
    #             region=base_irr.aoi_ee,
    #             image=joined_img,
    #             scale=30,
    #             assetId=base_asset_directory+'/joined_img123435',
    #             maxPixels=1e13
    #         )
    # task.start()
    # task = ee.batch.Export.image.toAsset(
    #             region=base_irr.aoi_ee,
    #             image=percent_similar,
    #             scale=30,
    #             assetId=base_asset_directory+'/testingTesting1237',
    #             maxPixels=1e13
    #         )
    # task.start()
    mean_val = percent_similar.reduceRegion(ee.Reducer.mean(), geometry = base_irr.aoi_ee, scale = 30)
    print("Mean val :", mean_val.getInfo())


def main():

    AOIs = [
    [0,0],                      # EMPTY
    [82.121452, 21.706688],     # C01 (India): 
    [-94.46643, 48.76297],      # C02 (Canada):  
    [-116.736866, 43.771114],   # C03 (Cent US - Idaho): 
    [10.640815, 52.185072],     # C04 (Germany): strange result... look at S2 layer! clouds?
    [10.7584699, 52.2058339],   # C05 (Germany): strange result
    [33.857852, 46.539389],     # C06 (Ukraine): good example, get second opinion
    [36.58565, 47.0838],        # C07 (Ukraine): Looks like color-labels are backwards. Notice how S2 is different from "Satellite" !!! 
    [38.34523, 30.22176],       # C08 (Saudi Arabia): Looks great, start with this!
    [-64.075199, -31.950112],   # C09 (Argentina): NEW
    [67.359826, 43.55412],      # C10 (Uzbekistan): mostly good
    [-46.2607, -11.93067],      # C11 (Brazil): strange result
    [76.07, 27.40],             # C12 (India): point from Sonal
    [76.812863, 20.248292],     # C13
    [76.833059, 20.325626],     # C14
    [-105.671054, 49.329958],
    [-101.86343, 48.10685],
    [10.7584699, 52.2058339],
    [10.723301, 52.145125],
    [38.38487, 30.20318],
    [58.463541, 42.573925],
    [-55.77832, -12.8049]       # C21
    ]

    nCases = 12

    # Either run this to create the files or to Evaluate (don't run single time to do both because it takes some time to create things)
    CREATE_FILES = False

    if CREATE_FILES:
        for CASE in range(1,nCases+1):
        # for CASE in range(1,2):
            base_aoi_lon = AOIs[CASE][0]
            base_aoi_lat = AOIs[CASE][1]
            aoi_edge_len = 0.05

            irr = irrigation30.irrigation30(center_lat=base_aoi_lat, center_lon=base_aoi_lon, edge_len=aoi_edge_len)

            # Create images as assets
            create_overlapping_images(irr, CASE)
            # the above takes some time -- may want to wait until the last task is finished

            # wait for keyboard input -- alert user to make sure the last one has been written

    if CREATE_FILES == False:
        for CASE in range(1,nCases+1):
        # for CASE in range(1,2):
            base_aoi_lon = AOIs[CASE][0]
            base_aoi_lat = AOIs[CASE][1]
            aoi_edge_len = 0.05

            base_irr = irrigation30.irrigation30(center_lat=base_aoi_lat, center_lon=base_aoi_lon, edge_len=aoi_edge_len)
            print('\nCase= ', CASE)
            evaluate_overlapping_images(3, base_irr, CASE)



if __name__ == '__main__':
    main()