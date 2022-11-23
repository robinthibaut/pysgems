import math
import os
import time
from os.path import join as join_path

import fiona
import numpy as np
import pandas as pd
import rasterio  # https://rasterio.readthedocs.io/en/latest/installation.html
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata
from geopandas import GeoDataFrame
from geopandas import points_from_xy
from loguru import logger
from matplotlib import pyplot as plt
from pyproj import Transformer


class PlotInterpolationMaps:
    """Class to plot the kriging results."""

    def __init__(self):
        ...

    def gslib(
        self,
        inputfile=str,
        dx_input=int,
        dy_input=int,
        x0_input=int,
        y0_input=int,
        x_lim_input=int,
        y_lim_input=int,
        outputfolder=str,
        typeKriging=str,
    ):
        """Function to add the coordinates to the gslib file that was the result of kriging.
        The results are saved as text files.

        :param inputfile: Path to the result data of kriging.
        :param dx_input: The x cell dimension of the grid.
        :param dy_input: The y cell dimension of the grid.
        :param x0_input: The x coordinate of the lower left corner of the grid.
        :param y0_input: The y coordinate of the lower left corner of the grid.
        :param x_lim_input: The x coordinate of the upper right corner of the grid.
        :param y_lim_input: The y coordinate of the upper right corner of the grid.
        :param outputfolder: Part of the path to the output file.
        :param typeKriging: The type of kriging.

        :return: None.
        """

        logger.info(f"Start adding the coordinates to the kriging results.")
        # get the inputfilename and the location of the output folder
        try:
            inputfile = inputfile
            inputfile = inputfile.replace("\\", "/")
            inputfile = inputfile.replace('"', "")
            inputfilename = inputfile.split("resultsdata2d")
            a = inputfilename[1]
            b = inputfilename[2]
            b = b.split("/")
            inputfilename = a + "resultsdata2d" + b[0] + b[1]
        except Exception:
            inputfile = inputfile
            inputfilename = inputfile.split("/")[-1].strip(".txt")
        outputfolder = outputfolder.replace("\\", "/")
        outputfolder = outputfolder.replace('"', "")

        # create the dimensions
        nx = (x_lim_input - x0_input) / dx_input  # number of cells in x direction
        ny = (y_lim_input - y0_input) / dy_input  # number of cells in y direction
        dx = dx_input  # the dimension of a cell in the x direction
        dy = dy_input  # the dimension of a cell in the y direction
        dz = 0  # the dimension of a cell in the z direction
        x0 = x0_input  # the x coordinate of the origin cell
        y0 = y0_input  # the y coordinate of the origin cell
        z0 = 0  # the z coordinate of the origin cell

        values = []

        # get the kriging values
        file = open(inputfile, "r")
        if typeKriging == "FIK":
            lines = file.readlines()[1:]
        elif typeKriging == "OK":
            lines = file.readlines()[3:]
        for line in lines:
            values.extend([line])

        # calculate the coordinates
        length = len(lines)
        ind = list(range(1, length + 1))
        coordinates = []
        for i in ind:
            iz = 1 + int((i - 1) / (nx * ny))  # get the coordinate of z
            iy = 1 + int((i - (iz - 1) * nx * ny) / nx)  # get the coordinate of y
            ix = i - (iz - 1) * nx * ny - (iy - 1) * nx  # get the coordinate of x
            if ix == 0:
                x = x0 + (dx / 2) + (nx - 1) * dx
                y = y
                z = z
                coordinate = [x, y, z]
                coordinates.extend([coordinate])
            else:
                x = x0 + (dx / 2) + (ix - 1) * dx
                y = y0 + (dy / 2) + (iy - 1) * dy
                z = z0 + (dz / 2) + (iz - 1) * dz
                coordinate = [x, y, z]
                coordinates.extend([coordinate])
        file.close()

        # add the coordinates to the kriging results
        ind2 = list(range(0, length))
        results = [["x", "y", "z", "value"]]
        for i in ind2:
            a = [float(values[i])]
            list1 = coordinates[i] + a
            results.extend([list1])
        np.savetxt(
            outputfolder + "/" + inputfilename + "-xyz_value.txt",
            results,
            delimiter="\t",
            fmt="%s",
        )
        logger.info(f"Saved the total results of kriging in the QGiS folder.")

        # Filter the results. Makes a new file without the -9966699.0 values.
        results_filtered = []
        for i in results[1:]:
            if i[3] == -9966699.0:
                None
            else:
                results_filtered.extend([i])
        if len(results) == len(results_filtered):
            None
        else:
            np.savetxt(
                outputfolder + inputfilename + "-xyz_value_filtered.txt",
                results_filtered,
                delimiter="\t",
                fmt="%s",
            )
            logger.info(f"Saved the filtered results of kriging in the QGiS folder.")

    def convert_latlon(self, x1=float, y1=float):
        """Convert the coordinates from one crs to another. Here Lambert72 to WGS84.
        :param x1: x input coordinate.
        :param y1: y input coordinate.
        :return: x2, y2: x and y output coordinates.
        """
        transformer = Transformer.from_crs("epsg:31370", "epsg:4326", always_xy=True)
        x2, y2 = transformer.transform(x1, y1)
        return x2, y2

    def mapping(
        self,
        inputdata=str,
        log_values=bool,
        name=str,
        levels_pK=None,
        indicator_levels=None,
        original_data=str,
        count=int,
        parameter=str,
        data_loc=str,
    ):
        """Create the interpolation raster and clip it to the Flanders region.
        Add the locations of some cities in Flanders.
        Add the locations of the datapoint to the map and in case of a value prediction map the datapoints can be
        plotted in the same color range as the map.
        It is possible to adjust the color range for the map.
        The maps for exceeding a threshold have a range between 0 and 1.
        This function also transforms the data back to non log values if necessary.
        The data from the variance result are also transformed to get the standard deviation values.

        :param inputdata: Path to the result data of kriging.
        :param log_values: If the values are log transformed.
        :param name: The name of the data column.
        :param levels_pK: The levels of the post kriging thresholds.
        :param indicator_levels: The levels of the indicator values.
        :param original_data: The path to the input data for kriging.
        :param count: The number of data columns.
        :param parameter: The parameter of the dataset.
        :param data_loc: The path to the datafile for kriging.
        :return: None.
        """
        logger.info(f"Start the creation of the interpolation raster plot for {name}.")
        name_plot = name
        df = pd.read_csv(inputdata, sep="\t")
        values = df["value"].values.tolist()
        name3 = name.split("_")[0]
        if log_values:
            if (name3 != "threshold") & (name3 != "level"):
                for j in range(len(values)):
                    values[j] = np.e ** values[j]
                df["value"] = values
        if (name == "ConditionalVariance") | (name == "kriging_krig_var"):
            for j in range(len(values)):
                values[j] = math.sqrt(values[j])
            df["value"] = values
        # df['lon'], df['lat'] = zip(*map(convert_latlon, df[0], df[1]))

        geometry = points_from_xy(x=df["x"], y=df["y"], crs="epsg:31370")
        df2 = GeoDataFrame(df, geometry=geometry)

        logger.info("Transform the data from vector to raster format.")
        geo_grid = make_geocube(
            vector_data=df2,
            measurements=["value"],
            resolution=(-100, 100),
            rasterize_function=rasterize_points_griddata,
        )
        logger.info("Plot the raster data.")
        geo_grid.value.where(geo_grid.value != geo_grid.value.rio.nodata).plot()
        name = inputdata.strip("-xyz_value.txt").strip("results_")
        geo_grid.rio.to_raster(raster_path=f"{name}_geo_grid.tif")
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        name = inputdata.strip("-xyz_value.txt").strip("results_")
        name2 = name.split("/")[-1]
        plt.title(f"{name2}")
        plt.imshow(geo_grid.value)
        plt.savefig(f"{name}_geo_grid.png")
        # plt.show()
        plt.clf()

        logger.info("Clip the raster data on the area of Flanders.")
        # Read Shape file
        cwd = os.getcwd()  # Working directory
        path_shp_flanders = join_path(
            cwd, "datasets", "demo_indicator_kriging", "shapefile", "flanders.shp"
        )
        with fiona.open(path_shp_flanders, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        # read imagery file
        with rasterio.open(f"{name}_geo_grid.tif") as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes)
            out_meta = src.meta
        # Save clipped imagery
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        with rasterio.open(
            f"{name}_geo_grid_imagery_trans_clip.tif", "w", **out_meta
        ) as dest:
            dest.write(out_image)

        dataset = rasterio.open(f"{name}_geo_grid_imagery_trans_clip.tif")
        image = dataset.read()

        fig, ax = plt.subplots(figsize=(15, 8))
        plt.xlabel("x coordinate (Lambert 72)")
        plt.ylabel("y coordinate (Lambert 72)")

        name2 = name_plot.split("_")[0]
        if (name2 == "threshold") | (name2 == "level"):
            if name3 == "threshold":
                index = name_plot.split("_")[1]
                level = levels_pK[int(index)]
                img = ax.imshow(image[0, :, :], extent=[20000, 260000, 150000, 250000])
                plt.title(f"Threshold {level} ng/l")
                img.set_clim(vmin=0, vmax=1)
                fig.colorbar(img, ax=ax)
                df = pd.read_csv(data_loc, sep="\t", skiprows=count + 5)
                for i in range(len(df)):
                    ax.plot(
                        float(df.iloc[:, 0][i]), float(df.iloc[:, 1][i]), "w.", ms=5
                    )
            if name2 == "level":
                index = name_plot.split("_")[1]
                level = indicator_levels[int(index)]
                img = ax.imshow(image[0, :, :], extent=[20000, 260000, 150000, 250000])
                plt.title(f"Indicator level {level} ng/l")
                img.set_clim(vmin=0, vmax=1)
                fig.colorbar(img, ax=ax)
                df = pd.read_csv(data_loc, sep="\t", skiprows=count + 5)
                for i in range(len(df)):
                    ax.plot(
                        float(df.iloc[:, 0][i]), float(df.iloc[:, 1][i]), "w.", ms=5
                    )

        if (name2 == "kriging") | (name2 == "ConditionalMean"):
            img = ax.imshow(image[0, :, :], extent=[20000, 260000, 150000, 250000])
            plt.title(f"Value prediction (ng/l)")
            img.set_clim(vmin=0, vmax=7)  # Define you your own color range.
            fig.colorbar(img, ax=ax)
            df = pd.read_csv(original_data, sep="\t")
            for i in range(len(df)):
                ax.scatter(
                    float(df.iloc[i]["x"].replace(",", ".")),
                    float(df.iloc[i]["y"].replace(",", ".")),
                    c=df.iloc[i][f"{parameter}"],
                    edgecolor="black",
                    vmin=0,
                    vmax=7,
                )  # add vmin and vmax with the same range as chosen above to plot the datapoint values within the
                # same range as the value prediction.
        if name2 == "kriging_krig_var":
            img = ax.imshow(image[0, :, :], extent=[20000, 260000, 150000, 250000])
            plt.title(f"Standard deviation prediction (ng/l)")
            img.set_clim(vmin=1.73, vmax=1.77)  # Define you your own color range.
            fig.colorbar(img, ax=ax)
            df = pd.read_csv(data_loc, sep="\t", skiprows=count + 5)
            for i in range(len(df)):
                ax.plot(float(df.iloc[:, 0][i]), float(df.iloc[:, 1][i]), "w.", ms=5)
        if name2 == "ConditionalVariance":
            img = ax.imshow(image[0, :, :], extent=[20000, 260000, 150000, 250000])
            plt.title(f"Standard deviation prediction (ng/l)")
            img.set_clim(vmin=1.5, vmax=2.2)  # Define you your own color range.
            fig.colorbar(img, ax=ax)
            df = pd.read_csv(data_loc, sep="\t", skiprows=count + 5)
            for i in range(len(df)):
                ax.plot(float(df.iloc[:, 0][i]), float(df.iloc[:, 1][i]), "w.", ms=5)

        # Plot the capital cities of each region of Flanders.
        ax.plot(70217.29, 211527.26, "k.", ms=5)
        ax.text(
            70217.29,
            211527.26,
            "Brugge",
            va="bottom",
            ha="left",
            color="black",
            fontweight="bold",
        )
        ax.plot(104868.65, 194024.25, "k.", ms=5)
        ax.text(
            104868.65,
            194024.25,
            "Gent",
            va="bottom",
            ha="left",
            color="black",
            fontweight="bold",
        )
        ax.plot(152132.21, 212370.17, "k.", ms=5)
        ax.text(
            152132.21,
            212370.17,
            "Antwerpen",
            va="bottom",
            ha="left",
            color="black",
            fontweight="bold",
        )
        ax.plot(218055.28, 180527.57, "k.", ms=5)
        ax.text(
            218055.28,
            180527.57,
            "Hasselt",
            va="bottom",
            ha="left",
            color="black",
            fontweight="bold",
        )
        ax.plot(173394.68, 174310.51, "k.", ms=5)
        ax.text(
            173394.68,
            174310.51,
            "Leuven",
            va="bottom",
            ha="left",
            color="black",
            fontweight="bold",
        )
        fig.savefig(f"{name}.png")
        plt.close("all")
        plt.clf()

    def main(
        self,
        inputpath=str,
        outputpath=str,
        dx_input=int,
        dy_input=int,
        x0_input=int,
        y0_input=int,
        x_lim_input=int,
        y_lim_input=int,
        log_values=bool,
        names=list,
        typekriging=str,
        levels_pK=None,
        indicator_levels=None,
        original_data=str,
        parameter=str,
        data_loc=str,
    ):
        """Call the functions to add the coordinates to the kriging result values and to create the interpolation raster.
        :param inputpath: Path to the result data of kriging.
        :param outputpath: Part of the path to the output file.
        :param dx_input: The x cell dimension of the grid.
        :param dy_input: The y cell dimension of the grid.
        :param x0_input: The x coordinate of the lower left corner of the grid.
        :param y0_input: The y coordinate of the lower left corner of the grid.
        :param x_lim_input: The x coordinate of the upper right corner of the grid.
        :param y_lim_input: The y coordinate of the upper right corner of the grid.
        :param log_values: If the values are log transformed.
        :param names: The name(s) of the data column(s).
        :param typekriging: The type of kriging.
        :param levels_pK: The levels of the post kriging thresholds.
        :param indicator_levels: The levels of the indicator values.
        :param original_data: The path to the input data for kriging.
        :param parameter: The parameter of the dataset.
        :param data_loc: The path to the datafile for kriging.
        :return: None.
        """

        start = time.time()

        df_results = pd.read_csv(
            inputpath, sep=" ", skiprows=len(names) + 2, header=None
        )
        df_results = df_results.iloc[:, :-1]
        df_results.columns = names
        outputfolder = outputpath
        for i in names:
            df_results[i].to_csv(
                f"{outputfolder}/results_{i}.txt", index=False, header=True
            )
            PlotInterpolationMaps.gslib(
                self,
                f"{outputfolder}/results_{i}.txt",
                dx_input,
                dy_input,
                x0_input,
                y0_input,
                x_lim_input,
                y_lim_input,
                outputfolder,
                typeKriging=typekriging,
            )
            if typekriging == "OK":
                count = len(names)
            if typekriging == "FIK":
                count = len(indicator_levels) - 1
            PlotInterpolationMaps.mapping(
                self,
                f"{outputfolder}/results_{i}-xyz_value.txt",
                log_values,
                i,
                levels_pK,
                indicator_levels,
                original_data,
                count=count,
                parameter=parameter,
                data_loc=data_loc,
            )

        logger.info(
            f"Ran the coordinates in {time.time() - start} s or {(time.time() - start) / 60} min"
        )
