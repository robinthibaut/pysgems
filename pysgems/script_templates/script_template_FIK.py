import os

import sgems

nodata = -9966699

os.chdir("RES_DIR")
sgems.execute("DeleteObjects computation_grid")
sgems.execute("DeleteObjects PROJECT_NAME")
sgems.execute("DeleteObjects finished")

for file in OBJECT_FILES:
    sgems.execute("LoadObjectFromFile  {}::All".format(file))
for parameter in range(len(PARAMETERS)):
    if parameter < (len(PARAMETERS) - 1):
        data = sgems.get_property("{}_grid".format(PARAMETERS[parameter + 1]),"{}".format(PARAMETERS[parameter + 1]))
        sgems.set_property("{}_grid".format(PARAMETERS[parameter]),"{}".format(PARAMETERS[parameter + 1]),data)
        sgems.execute("DeleteObjects {}_grid".format(PARAMETERS[parameter + 1]))

sgems.execute("NewCartesianGrid  computation_grid::GRID")
sgems.execute('RunGeostatAlgorithm  indicator_kriging::/GeostatParamUtils/XML::ALGORITHM_XML1')
sgems.execute('RunGeostatAlgorithm  PostKriging::/GeostatParamUtils/XML::ALGORITHM_XML2')
sgems.execute('SaveGeostatGrid  computation_grid::FEATURE_OUTPUT.grid::gslib::0::OUTPUT_LIST')
