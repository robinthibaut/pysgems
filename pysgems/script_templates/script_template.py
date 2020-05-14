import os

import sgems

nodata = -9966699

os.chdir("RES_DIR")
sgems.execute('DeleteObjects computation_grid')
sgems.execute('DeleteObjects PROJECT_NAME')
sgems.execute('DeleteObjects finished')

for file in OBJECT_FILES:
    sgems.execute('LoadObjectFromFile  {}::All'.format(file))

sgems.execute('NewCartesianGrid  computation_grid::GRID')

#~sgems.execute('RunGeostatAlgorithm  ALGORITHM_NAME::/GeostatParamUtils/XML::ALGORITHM_XML')
#~sgems.execute('SaveGeostatGrid  computation_grid::FEATURE_OUTPUT.grid::gslib::0::OUTPUT_LIST')