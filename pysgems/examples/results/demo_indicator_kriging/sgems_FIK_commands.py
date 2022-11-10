import os

import sgems

nodata = -9966699

os.chdir(
    "C://Users//guill//Documents//UGent//2021-2022//Thesis//pysgems//pysgems//examples//results//demo_indicator_kriging"
)
sgems.execute("DeleteObjects computation_grid")
sgems.execute("DeleteObjects sgems_FIK")
sgems.execute("DeleteObjects finished")

for file in [
    "level_0.sgems",
    "level_1.sgems",
    "level_2.sgems",
    "level_3.sgems",
    "level_4.sgems",
    "level_5.sgems",
    "level_6.sgems",
    "level_7.sgems",
]:
    sgems.execute("LoadObjectFromFile  {}::All".format(file))
for parameter in range(
    len(
        [
            "level_0",
            "level_1",
            "level_2",
            "level_3",
            "level_4",
            "level_5",
            "level_6",
            "level_7",
        ]
    )
):
    if parameter < (
        len(
            [
                "level_0",
                "level_1",
                "level_2",
                "level_3",
                "level_4",
                "level_5",
                "level_6",
                "level_7",
            ]
        )
        - 1
    ):
        data = sgems.get_property(
            "{}_grid".format(
                [
                    "level_0",
                    "level_1",
                    "level_2",
                    "level_3",
                    "level_4",
                    "level_5",
                    "level_6",
                    "level_7",
                ][parameter + 1]
            ),
            "{}".format(
                [
                    "level_0",
                    "level_1",
                    "level_2",
                    "level_3",
                    "level_4",
                    "level_5",
                    "level_6",
                    "level_7",
                ][parameter + 1]
            ),
        )
        sgems.set_property(
            "{}_grid".format(
                [
                    "level_0",
                    "level_1",
                    "level_2",
                    "level_3",
                    "level_4",
                    "level_5",
                    "level_6",
                    "level_7",
                ][parameter]
            ),
            "{}".format(
                [
                    "level_0",
                    "level_1",
                    "level_2",
                    "level_3",
                    "level_4",
                    "level_5",
                    "level_6",
                    "level_7",
                ][parameter + 1]
            ),
            data,
        )
        sgems.execute(
            "DeleteObjects {}_grid".format(
                [
                    "level_0",
                    "level_1",
                    "level_2",
                    "level_3",
                    "level_4",
                    "level_5",
                    "level_6",
                    "level_7",
                ][parameter + 1]
            )
        )

sgems.execute(
    "NewCartesianGrid  computation_grid::2400::1000::1::100::100::0::20000::150000::0.0"
)
sgems.execute(
    'RunGeostatAlgorithm  indicator_kriging::/GeostatParamUtils/XML::<parameters><algorithm name="indicator_kriging" /><Hard_Data_Grid value="level_0_grid" region="" /><Hard_Data_Property count="8" value="level_0;level_1;level_2;level_3;level_4;level_5;level_6;level_7" /><Min_Conditioning_Data value="0" /><Max_Conditioning_Data value="12" /><Search_Ellipsoid value="200000 200000 0 0 0 0" /><AdvancedSearch use_advanced_search="0" /><Grid_Name value="computation_grid" region="" /><Property_Name value="indicator_kriging" /><Nb_Realizations value="0" /><Nb_Indicators value="8" /><Categorical_Variable_Flag value="0" /><Marginal_Probabilities value="0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95" /><Median_Ik_Flag value="0" /><Full_Ik_Flag value="1" /><Variogram_Full_Ik nugget="0.208817791223917" structures_count="1"><structure_1 contribution="0.0345295482171543" type="Spherical"><ranges max="64982.85891440897" medium="64982.85891440897" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik><Variogram_Full_Ik_2 nugget="0.185006575212523" structures_count="1"><structure_1 contribution="0.0427652898836473" type="Spherical"><ranges max="54046.57177403155" medium="54046.57177403155" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik_2><Variogram_Full_Ik_3 nugget="0.1661667836255214" structures_count="1"><structure_1 contribution="0.0517441692331019" type="Spherical"><ranges max="49277.431784563945" medium="49277.431784563945" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik_3><Variogram_Full_Ik_4 nugget="0.1661667836255214" structures_count="1"><structure_1 contribution="0.0517441692331019" type="Spherical"><ranges max="49277.431784563945" medium="49277.431784563945" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik_4><Variogram_Full_Ik_5 nugget="0.1428386652732374" structures_count="1"><structure_1 contribution="0.0239039585849443" type="Spherical"><ranges max="54699.61433903228" medium="54699.61433903228" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik_5><Variogram_Full_Ik_6 nugget="0.1104858895731466" structures_count="1"><structure_1 contribution="0.0244660689593586" type="Spherical"><ranges max="44290.812895819945" medium="44290.812895819945" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik_6><Variogram_Full_Ik_7 nugget="0.0968901925036019" structures_count="1"><structure_1 contribution="6.380590500398853e-13" type="Spherical"><ranges max="67188.53075966128" medium="67188.53075966128" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik_7><Variogram_Full_Ik_8 nugget="0.050214843020313586" structures_count="1"><structure_1 contribution="6.905032101656161e-13" type="Spherical"><ranges max="80080.54842994284" medium="80080.54842994284" min="0" /><angles x="0" y="0" z="0" /></structure_1></Variogram_Full_Ik_8></parameters>'
)
sgems.execute(
    'RunGeostatAlgorithm  PostKriging::/GeostatParamUtils/XML::<parameters> <algorithm name="PostKriging" /> <Hard_Data value="computation_grid" region="" /><is_non_param_cdf value="1" /><is_gaussian value="0" /><props count="8" value="indicator_kriging__real0;indicator_kriging__real1;indicator_kriging__real2;indicator_kriging__real3;indicator_kriging__real4;indicator_kriging__real5;indicator_kriging__real6;indicator_kriging__real7" /><marginals value="0.6931471805599453 1.0986122886681098 1.3862943611198906 1.6094379124341003 1.8534197411589592 2.4022458416478507 2.616824937969143 2.9092952853407605" /><lowerTailCdf function="Power" extreme="0" omega="2.5" /><upperTailCdf function="Hyperbolic" extreme="0" omega="1.5" /><mean value="1" /><cond_var value="1" /><cond_var_prop value="conditional_variance_FIK" /><iqr value="0" /><quantile value="0" /><prob_above value="1" /><prob_above_vals value="0.0 0.6931471805599453 1.0986122886681098 1.3862943611198906 1.6094379124341003 1.791759469228055 1.9459101490553132 2.0794415416798357 2.1972245773362196 2.302585092994046 2.995732273553991" /><prob_above_prop value = "threshold_level" /><prob_below value="0" /><mean_prop value="conditional_mean_FIK" /></parameters>'
)
sgems.execute("SaveGeostatGrid  computation_grid::results.grid::gslib::0::")
