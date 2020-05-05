import sgems
import os

os.chdir("RES_DIR")
sgems.execute('DeleteObjects ALGORITHM_NAME')
sgems.execute('DeleteObjects PROJECT_NAME')
sgems.execute('DeleteObjects finished')

sgems.execute('NewCartesianGrid  computation_grid::GRID')

properties = FEATURES_LIST
nodata = -9966699
sgems.execute('NewCartesianGrid  PROJECT_NAME::GRID')

nrow, ncol = list(map(int,'GRID'.split('::')[:2]))

with open("NODES_VALUES_FILE") as nf:
    fn = eval(nf.read())
for p in range(len(properties)):
    hard_data = [nodata for i in range(int(nrow*ncol))]
    for n in fn[p]:
        hard_data[int(n[0])] = n[1]
    sgems.set_property('PROJECT_NAME', properties[p], hard_data)
    
sgems.execute('RunGeostatAlgorithm  ALGORITHM_NAME::/GeostatParamUtils/XML::ALGORITHM_XML')

sgems.execute('SaveGeostatGrid  computation_grid::FEATURE_OUTPUT.grid::gslib::0::PROPERTY_NAME')

