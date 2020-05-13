import os

import sgems

os.chdir("RES_DIR")
sgems.execute('DeleteObjects computation_grid')
sgems.execute('DeleteObjects PROJECT_NAME')
sgems.execute('DeleteObjects finished')

for file in OBJECT_FILES:
    sgems.execute('LoadObjectFromFile  {}::All'.format(file))

sgems.execute('NewCartesianGrid  computation_grid::GRID')

try:
    with open("NODES_VALUES_FILE") as nf:
        fn = eval(nf.read())

    properties = FEATURES_LIST
    nodata = -9966699
    sgems.execute('NewCartesianGrid  hard_data::GRID')
    nrow, ncol = list(map(int, 'GRID'.split('::')[:2]))

    for p in range(len(properties)):
        hard_data = [nodata for i in range(int(nrow*ncol))]
        for n in fn[p]:
            hard_data[int(n[0])] = n[1]
        sgems.set_property('PROJECT_NAME', properties[p], hard_data)
except IOError:
    pass

#~sgems.execute('RunGeostatAlgorithm  ALGORITHM_NAME::/GeostatParamUtils/XML::ALGORITHM_XML')
#~sgems.execute('SaveGeostatGrid  computation_grid::FEATURE_OUTPUT.grid::gslib::0::PROPERTY_NAME')