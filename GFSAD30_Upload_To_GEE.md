# Importing GFSAD30 data to GEE

##### GFSAD30 Details
GFSAD30 is single-band data of 30m pixel GeoTIFF tiles, with the following band encoding:
* 0: Water
* 1: Non-Cropland
* 2: Cropland
* 3: User-defined no-data value

##### NASA EARTHDATA Credentials
Login credentials are required to download data. Sign up @ https://urs.earthdata.nasa.gov

##### Procedure Notes:
1. All steps verified on macOS Catalina with bash shell.
2. Process will download 459 tiles and consume 7.9 Gb of local drive space.
3. Activity will be within GEE Asset limits of:
* 10 Gb individual GeoTIFF files
* 250 Gb total
* 10k files
##### Acquiring Download Scripts

1. Begin @ https://croplands.org/downloadLPDAAC

Repeat below step 2-10 for each desired geography
2. Choose `Download` link for each geography
3. Choose `Access Data` link
4. Choose download arrow for `NASA Earthdata Search`
5. Choose result
6. Choose `Download All`
7. Choose `Direct Download` and `Done`
8. Choose `Download Data`
9. Choose `Download Access Script`
10. Choose `Download Script`

##### Executing Scripts
1. Move all scripts to desired folder.
Repeat steps 2-3 for all scripts
2. Terminal: Make script executable `chmod 777 <scriptname.sh>`
3. Terminal: Execute script `./<scriptname.sh>` You will be prompted to enter your NASA WWHHHAT username and password. 

##### Uploading GFSAD30 Image to GEE 
Strategy is to upload all downloaded tiles simultaneously to GEE to create a single mosaic image.
1. `GEE` > `Assets` > `NEW` > `GeoTiff`.
2. Drag/Drop all 459 files to `SELECT` button.
3. Choose `Asset Name` for mosaic image.
4. `Advanced Options` > `Masking mode` > `No-data value` > Enter `3`. Note that this will code any instances of non-data as `3`. 

##### Sharing Image
1. Choose `<Asset Name>`
2. Add `<user's GEE login email address>` > `Done`

##### Reviewing File in GEE Viewer
1. Run below script. 
```
    var myimg = ee.Image("users/<username>/<asset name>");
    Map.addLayer(myimg);
```
2. Click on test points in map & view `b1` value for points
