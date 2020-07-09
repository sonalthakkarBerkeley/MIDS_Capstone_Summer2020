def Create_AOI_Box(center_lat,center_lon,edge_len):
'''
	This function creates the GEE bounding box centered at the coordinate given with the edge length given
'''
	return ee.Geometry.Rectangle([center_lon-edge_len/2, center_lat-edge_len/2, 
												center_lon+edge_len/2, center_lat+edge_len/2])

def Model_Pipeline_Placeholder(aoi):
    # will be the actual pipeline in the future but for now will load from pickle and shift
    new_df = pd.read_pickle('simple_df')
    return new_df

def Evaluate_Consistency(aoi_lat, aoi_lon, aoi_edge)
'''
	This function will determine the consistency of output for each pixel in the bounding box
	defined by the cenroid of aoi_lat,aoi_lon with edges of size aoi_edge.
	Consistency is defined as the percentage of times each pixel was labeled the same way.


	Evaluate_Consistency will run nSegments ^ 2 iterations of the model and product an
	Inputs: 
		* Bounded Region to evaluate
			* aoi_lat, aoi_lon, aoi_edge (center point)
	Output:
		* [dataframe, median(% consistent), mean(% consistent)]
		* 1 dataframe:
			* # Positive Cases (Irrigated) [Lat,Lon,# Positive, # Negative, % Consistent]
				Note % Consistent defined as max(% positive, % negative)

	Assumptions:
		* Hardcoding nSegments = 5. Logic would be slightly different for even values. Future
		* Assuming already ran ee.Authenticate(), ee.Initialize()
		* Format of Model_Pipeline dataset will be pandas dataframe with columns: [lat,lon,prediction]

	Note that with 5 Segments the underlying model will be called 25 times (5*5). 
	There will be vertical (latitude shifts) and horizontal (longitude shifts).
	The middle value would be the main area_of_interest while the others correspond to shifts

'''
	# Assuming already ran ee.Authenticate(), ee.Initialize()
	# Hard-Coded Values (for now): 
	nSegments = 5 # would like this value to remain odd
	aoi_lat, aoi_lon = 43.771114, -116.736866
	aoi_edge_len = 0.005

	area_of_interest_main = Create_AOI_Box(aoi_lat, aoi_lon, aoi_edge_len)

	step_size = edge_len / nSegments
	(i - (math.floor(nSegments / 2)))
	latitude_vals = [aoi_lat + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]
	longitude_vals = [aoi_lon + step_size * (i-math.floor(nSegments / 2)) for i in range(nSegments)]

	# Build Base Table -- will add to this through iterations
	base_predictions = Model_Pipeline_Placeholder(area_of_interest_main)
	base_predictions['Pos_Case_Count'] = base_predictions['mod']
	base_predictions['Neg_Case_Count'] = 1-base_predictions['mod']
	base_predictions.drop(columns=['mod'], inplace=True)

	for lat in latitude_vals:
		for lon in longitude_vals:
			# Don't rerun base case because we built that to begin with (above)
			if lat == aoi_lat and lon == aoi_lon:
				continue
			else:
				# join new predictions (new lat/lon) in with base predictions and increment counts
				# making assumption that join will work (output from GEE will have consistent Lat/Lon)
				# otherwise may need to round/truncate at some decimal point
				new_predictions = Model_Pipeline_Placeholder(Create_AOI_Box(lat, lon, aoi_edge_len))
				base_predictions = base_predictions.merge(new_predictions, how='left', on=['lat','lon'])
				base_predictions['Pos_Case_Count'] += base_predictions['mod']
				base_predictions['Neg_Case_Count'] += 1-base_predictions['mod']
				base_predictions.drop(columns=['mod'], inplace=True)


	# Now can do calculation of how consistent we have been across the above
	base_predictions['Pct_Consistent'] = base_predictions.apply(lambda row: 
											max(row['Pos_Case_Count']/(row['Pos_Case_Count'] + \
	                                                      row['Neg_Case_Count']),
				                               row['Neg_Case_Count']/(row['Pos_Case_Count'] + \
				                                                      row['Neg_Case_Count']) )
											, axis=1)

	return (base_predictions, 
			base_predictions.Pct_Consistent.median(),
			base_predictions.Pct_Consistent.mean())



