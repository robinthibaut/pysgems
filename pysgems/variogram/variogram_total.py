# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from decimal import Decimal
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import scipy.optimize as optimize
from scipy.special import kv,gamma # gamma needed for expansion variogramFun
from itertools import combinations
import warnings
import os
from os.path import join as jp
from loguru import logger
import time

outputfile = os.getcwd()
outputfile = jp(outputfile, "variogram")


def main(inputfile,nbpoints):
    start = time.time()
    columnx = 0
    columny = 1
    columnAs = 2

    columnx = int(columnx)
    columny = int(columny)
    columnAs = int(columnAs)

    inputfile = inputfile.replace("\\", "/")
    inputfile = inputfile.replace('"', '')


    def read_col(fname, col, convert=str, sep='\t'):
        with open(fname) as fobj:
             return [convert(line.split(sep=sep)[col]) for line in fobj]

    x = read_col(inputfile, columnx)
    y = read_col(inputfile, columny)
    As = read_col(inputfile, columnAs)

    listx = []
    for i in x:
        if i[-2:] == '.0':
            value = i[:-2]
            i = value
            listx.extend([i])
        else:
            listx.extend([i])
    #print(listx)

    listy = []
    for i in y:
        if i[-2:] == '.0':
            value = i[:-2]
            i = value
            listy.extend([i])
        else:
            listy.extend([i])
    #print(listy)

    listAs = []
    for i in As:
        if i[-2:] == '.0':
            value = i[:-2]
            i = value
            listAs.extend([i])
        else:
            listAs.extend([i])
    #print(listAs)


    A2 = [float(a.strip('"')) for a in listx]
    #print(A2)


    B2 = [float(b.strip('"')) for b in listy]
    #print(B2)

    C2 = [float(c.strip('"')) for c in listAs]
    #print(C2)


    coords = list(zip(A2, B2))
    #print(coords)


    coords = np.array(coords)
    #print(coords)
    #np.info(coords)

    values = C2
    values = np.array(values)
    values.shape = (len(values), 1)
    #print(values)
    #np.info(values)
    listvalues = []
    listdistances = []
    listnr_pairs = []
    def variogram(coords, values, weights=None, max_dist=None,
                  lag_type='equalSize', nr_lags=nbpoints, lag_edges=None,
                  poisson=False, anisotropy=False, theta_step=90,
                  var_type='gamma', plotit=True,
                  subsample=None):
        """
        isotropic and anisotropic experimental (semi-)variogram

        Parameters
        ----------
        coords : array (n-by-2)
            Array with coordinates. Each row is a location in a
            2-dimensional space (e.g. [x y]).
        values : array (n-by-1)
            Column array with values of the locations in x.
        weights : array (n-by-1), optional
            Column array with population weights of the locations in x.
            The default is None (no weights).
        max_dist : scalar, optional
            Maximum distance for variogram calculation.
            The default is None (maximum distance in the dataset / 2).
        lag_type : string, optional
            Calculation method for the lags:
            'equalSize' (default), 'equalPairs', 'userDefined'.
        nr_lags : scalar, optional
            Number lag classes the distance should be grouped into.
            The default is 20.
        lag_edges : array (k), optional
            Array with the edges of the user-defined lags. This input has only an
            effect when the lag type is set to 'userDefined'.
            The default is None.
        poisson : boolean, optional
            Calculate the experimental variogram for poisson kriging
            (weight vector required). The default is False.
        anisotropy : boolean, optional
            Calculate the experimental variogram for different directions.
            The default is False.
        theta_step : scalar, optional
            Angle between the directional axis in degrees. The default is 45.
        var_type : string, optional
            'gamma' returns the variogram value (default)
            'cloud1' returns the binned variogram cloud
            'cloud2' returns the variogram cloud
        plotit : boolean, optional
            Plot the final outcome. The default is False.
        subsample : scalar, optional
            Number of randomly drawn points if large datasets are used.
            scalar (positive integer, e.g. 3000)
            or a fraction (e.g. 0.2 resulting a subsample of 1000/5000)
            None (default = no subsampling).

        Returns
        -------
        values : array (k-by-l)
            Calculated variogram values.
        distances : array (k-by-1)
            The distances of the corresponding variogram values.
        theta : array (1-by-l)
            Angles of the correspond variogram values.
            (only calculated for a directional variogram)
        nr_pairs : array (k-by-l)
            Number of point combinations grouped in the corresponding lag class.
            (not calculated for var_type 'cloud2')

        """

        ## Check inputs and set defaults
        # Concatenate column vectors
        if weights is None:
            data = np.hstack((coords, values))
        else:
            data = np.hstack((coords, values, weights))

        # Remove rows with nan (points, values or weights)
        data = data[~np.all(np.isnan(data), axis=1), :]
        nr_data = len(data)

        # Set default max_dist as the radius of the coordinates bounding box
        max_dist_def = pdist(np.array([[data[:, 0].min(), data[:, 1].min()],
                                       [data[:, 0].max(), data[:, 1].max()]])) / 2
        if max_dist is None:
            max_dist = max_dist_def
        elif max_dist > max_dist_def:
            warnings.warn('Warning: The maximum distance for variogram ' +
                          'calculation exceeds the radius of the bounding box ' +
                          'of the study area.')

        # Check lag_type
        lag_type = lag_type.lower()
        if isinstance(lag_type, str):
            if not (lag_type == 'equalsize'
                    or lag_type == 'equalpairs'
                    or lag_type == 'userdefined'
                    or lag_type == 'equalsize'):
                raise ValueError('lag_type is not recognised, please ' +
                                 'provide a valid name.')
        else:
            raise ValueError('lag_type has to be a string.')

        # Check nr_lags
        if not isinstance(nr_lags, int) or nr_lags < 1:
            raise ValueError('nr_lags is not valid, please provide ' +
                             'an integer larger than 0.')

        # Check lag_edges
        if lag_edges is not None:
            lag_type = 'userdefined'  # Redefine lag_type
            if (len(lag_edges) < 2
                    or np.min(lag_edges) < 0
                    or np.max(lag_edges) > max_dist_def
                    or max(lag_edges) > max_dist):
                raise ValueError('lag_edges is not valid, make sure that ' +
                                 'atleast 2 valid edges are given and that all ' +
                                 'edges are between 0 and max_dist.')

        # Check poisson
        if poisson and weights is None:
            poisson = False
            warnings.warn('Warning: the variogram estimated for poisson ' +
                          'kriging, can only be calculated when weights are ' +
                          'defined. The calculation is continued for weighted ' +
                          'variograms.')

        # Check anisotropy
        if anisotropy and not np.shape(coords)[1] == 2:
            anisotropy = False
            warnings.warn('Warning: Anisotropy is only supported for 2D data.')

        # Check theta_step
        if not isinstance(theta_step, int) or theta_step < 1 or theta_step > 90:
            raise ValueError('theta_step is not valid, please provide ' +
                             'an integer larger than 0 and smaller then 90.')

        # Check var_type
        if isinstance(var_type, str):
            var_type = var_type.lower()
            if not (var_type == 'gamma'
                    or var_type == 'cloud1'
                    or var_type == 'cloud2'):
                raise ValueError('var_type is not recognised, please ' +
                                 'provide a valid name.')
        else:
            raise ValueError('var_type has to be a string.')

        # Extract a random subset of the data
        if subsample is not None:
            if subsample > 1 and subsample <= nr_data:
                I = np.random.random_integers(1, nr_data, subsample)
            elif subsample > 0 and subsample < 1:
                I = np.random.randint(1, nr_data, round(nr_data * subsample))
            data = data[I, :]
            nr_data = len(data)

        # Calculate condensed distances matrix (start point, end point, distance)
        distances = np.empty((nr_data * (nr_data - 1) // 2, 3))
        for i, comb in enumerate(combinations(range(nr_data), 2)):
            distances[i, 0:2] = comb
        distances[:, 2] = pdist(data[:, 0:2])

        # Retain combinations (rows) that are within 0 < x <= max_dist
        distances = distances[(distances[:, 2] > 0)
                              & (distances[:, 2] <= max_dist), :]

        # Calculate the lag edges
        if lag_type == 'equalsize':
            lag_edges = np.linspace(0, max_dist, nr_lags + 1).flatten()
            i_lags = np.digitize(distances[:, -1], lag_edges, right=True) - 1
        elif lag_type == 'equalpairs':
            i_lags = np.repeat(np.arange(0, nr_lags), nr_data // nr_lags)
            nr_pairs = len(i_lags)

            distances = distances[np.argsort(distances[:, -1]), :]
            distances = distances[0:nr_pairs - 1, :]

            i_edges = np.argwhere(i_lags != np.roll(i_lags, -1))
            lag_edges = np.array([0, distances[i_edges, -1]]).flatten()
        elif lag_type == 'userdefined':
            lag_edges = lag_edges.flatten()
            i_lags = np.digitize(distances[:, -1], lag_edges, right=True) - 1

        # Indices of all combinations (start and end)
        i_combs = distances[:, 0:2].astype(int, casting='unsafe')

        # Calculate squared difference between values of the pairs
        values_sqd = np.diff(data[i_combs, 2], axis=1) ** 2

        # Calculate the weights
        if poisson:
            # weights(tail) * weights(head)/(weights(tail) + weights(head))
            w = (data[i_combs[:, 0], 3] * data[i_combs[:, 1], 3]) \
                / (data[i_combs[:, 0], 3] + data[i_combs[:, 1], 3])
            w = w.reshape(-1, 1)

            # sum(weights*values) / sum(weights)
            m_star = (data[:, 2] * data[:, 3]).sum() / data[:, 3].sum()
        elif weights is not None:
            # weights(tail) * weights(head)
            w = data[i_combs[:, 0], 3] * data[i_combs[:, 1], 3]
            w = w.reshape(-1, 1)

        # Prepare anisotropy variables
        if anisotropy:
            nr_theta_edges = 180 // theta_step + 1

            # Convert to radians
            theta_step = theta_step / 180 * np.pi

            # Calculate the angles (clockwise starting at '12h'/North = 0 rad) by
            # inverting the  x and y coordinates in np.arctan2(y,x) to
            # np.arctan2(x,y).
            # Only a semicirle is needed, other half is mirrored over the origin.
            theta = np.arctan2(coords[i_combs[:, 1], 0]
                               - coords[i_combs[:, 0], 0],
                               coords[i_combs[:, 1], 1]
                               - coords[i_combs[:, 0], 1])
            theta[theta < 0] = theta[theta < 0] + np.pi
            theta[theta >= (np.pi - theta_step / 2)] = 0

            # Binning of theta (0 .. 180 degrees)
            theta_edges = np.linspace(-theta_step / 2, np.pi - theta_step / 2,
                                      nr_theta_edges)
            i_theta = np.digitize(theta, theta_edges) - 1

            # Bin centers
            theta_centr = theta_edges[:-1] + theta_step / 2

        # Calculate the variogram
        if var_type == 'gamma':
            # Define specific variogram function
            if poisson:
                # Poisson variogram function
                var_function = lambda x: (1. / (2 * np.sum(w[x])) \
                                          * np.sum(w[x] * values_sqd[x] - m_star))
            elif weights is not None:
                # Weighted variogram function
                var_function = lambda x: (1. / (2 * np.sum(w[x])) \
                                          * np.sum(w[x] * values_sqd[x]))
            else:
                # Default variogram function
                var_function = lambda x: (1. / (2 * np.sum(x))
                                          * np.sum(values_sqd[x]))

            if anisotropy:
                vg_distances = np.convolve(lag_edges, np.ones(2), 'valid') / 2
                vg_distances = distances.reshape((-1, 1))
                vg_theta = theta_centr.reshape((1, -1))
                vg_values = np.zeros((nr_lags, len(theta_centr)))
                vg_nr_pairs = np.zeros((nr_lags, len(theta_centr)))

                for k in range(nr_lags):
                    for l in range(len(theta_centr)):
                        I = (i_lags == k) & (i_theta == l)

                        vg_values[k, l] = var_function(I)

                        vg_nr_pairs[k, l] = np.sum(I)

            else:
                vg_distances = np.convolve(lag_edges, np.ones(2), 'valid') / 2
                vg_values = np.zeros((nr_lags, 1))
                vg_nr_pairs = np.zeros((nr_lags, 1))

                for k in range(nr_lags):
                    I = i_lags == k

                    vg_values[k] = var_function(I)
                    vg_nr_pairs[k] = np.sum(I)
                vg_values = vg_values.flatten()
                vg_nr_pairs = vg_nr_pairs.flatten()

        elif var_type == 'cloud1':
            vg_distances = np.convolve(lag_edges, np.ones(2), 'valid') / 2
            vg_distances = vg_distances[i_lags]
            vg_values = values_sqd.flatten()
            if anisotropy:
                vg_theta = theta_centr[i_theta].flatten()
        elif var_type == 'cloud2':
            vg_distances = distances[:, -1]
            vg_values = values_sqd.flatten()
            if anisotropy:
                vg_theta = theta_centr[i_theta].flatten()

        # Create plot
        if plotit & anisotropy & (var_type != 'gamma'):
            warnings.warn('Warning: An anisotropic cloud plot can not be displayed.')

        elif plotit:
            if var_type == 'gamma':
                marker = 'bo--'
            else:
                marker = '.'

            if anisotropy:
                # Adjust thetas to account for full circle
                p_theta, p_r = np.meshgrid(
                    np.concatenate([theta_edges,
                                    theta_edges[1:] + np.pi]),
                    lag_edges)
                p_values = np.tile(vg_values, (1, 2))

                ax = plt.subplot(1, 1, 1, projection='polar')
                p_var = ax.pcolormesh(p_theta, p_r, p_values, cmap="Reds")

                ax.set_title('Anisotropic variogram')

                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)

                cbar = plt.colorbar(p_var)
                cbar.ax.set_ylabel('Variogram value ($\gamma(h)$)')

                ax.set_rmax(max_dist)
                ax.set_yticks(lag_edges)
                ax.set_yticklabels(np.round(lag_edges))
                ax.set_rlabel_position(0)

                plt.show()
            else:
                plt.plot(vg_distances, vg_values, marker)
                plt.xlabel('Lag distance (h)')
                plt.ylabel('Variogram value ($\gamma(h))$')
                plt.ylim(0)
                plt.title('Variogram')

                plt.show()

        # Return outputs
        if var_type == 'cloud2':
            #print("vg_values = ", vg_values, "\n", "vg_distances = ", vg_distances, "\n", "\n")
            return vg_values, vg_distances

        else:
            if anisotropy:
              #  print("vg_values = ", vg_values, "\n", "vg_distances = ", vg_distances, "\n", "vg_theta = ", vg_theta, "\n", "vg_nr_pairs =", vg_nr_pairs, "\n", "\n")
                return vg_values, vg_distances, vg_theta, vg_nr_pairs
            else:
               # print("vg_values = ", vg_values, "\n", "vg_distances = ", vg_distances, "\n", "vg_nr_pairs =", vg_nr_pairs, "\n", "\n")
                listvalues.extend(vg_values)
                listdistances.extend(vg_distances)
                listnr_pairs.extend(vg_nr_pairs)
                #print("listvalues = ", listvalues)
                #print("listdistances = ", listdistances)
                #print("listnr_pairs = ", listnr_pairs)
                return vg_values, vg_distances, vg_nr_pairs

    variogram(coords, values)

    values = np.array(listvalues)
    values.shape = (len(values), 1)
    #print(values)
    distances = np.array(listdistances)
    distances.shape = (len(distances), 1)
    #print(distances)
    nr_pairs = np.array(listnr_pairs)
    nr_pairs.shape = (len(nr_pairs), 1)
    #print(nr_pairs)

    vg_range = input("What is the range? ")
    vg_sill = input("What is the sill? ")
    vg_nugget = input("What is the nugget? ")
    vg_range = float(vg_range)
    vg_sill = float(vg_sill)
    vg_nugget = float(vg_nugget)

    model = input('Which model does best fit the data? (spherical/exponential/gaussian) ')
    listvar = []
    def variogramfit(distances, values, nr_pairs=None,
                     vg_range=[], vg_sill=[], vg_nugget=[], vg_model=model,
                     weightfun='none', plotit=True, full=True):
        """
        fit a theoretical variogram to an experimental variogram

        variogramfit performs a (weighted) least squares fit of various
        theoretical variograms to an experimental, isotropic variogram. The
        user can choose between various bounded (e.g. spherical) and unbounded
        (e.g. exponential) models. A nugget variance and higher nested models
        can be modelled as well.

        Nested variogram models can be defined by including multiple initial
        values (i.e. in case a double exponential model is targeted, a list of
        two range elements, two sill elements and two model types is inserted).
        Only one initial nugget values needs to be applied.

        The variogram fitting algorithm is in particular sensitive to initial
        values below the optimal solution. In case you have no idea of
        initial values variogramfit calculates initial values for you
        (c0 = max(gammaexp); a0 = max(h)*2/3;). If this is a reasonable
        guess remains to be answered. Hence, visually inspecting your data
        and estimating a theoretical variogram by hand should always be
        your first choice.

        Note that for unbounded models, the supplied parameter a0 (range) is
        the distance where gamma equals 95 % of the sill variance. The
        returned parameter a0, however, is the parameter r in the model. The
        range at 95 % of the sill variance is then approximately 3*r.

        Based on code of Wolfgang Schwanghart (w.schwanghart[at]unibas.ch)

        Parameters
        ----------
        distances : array (1-by-k)
            The distances of the experimental variogram lag classes.
            Calculated according to 'variogram'.
        values : array (1-by-k)
            Calculated experimental variogram values.
            Calculated according to 'variogram'.
        nr_pairs : array (1-by-k), optional
            Number of point combinations grouped in the corresponding lag class.
            Calculated according to 'variogram'.
            Necessary for the calculation of the weight functions 'cressie85'
            and 'mcbratney86'.
            The default is None.
        vg_range : scalar or list, optional
            Initial estimate of the range(s) of the variogram model.
            The default is max(distances)*2/3.
        vg_sill : scalar or list, optional
            Initial estimate of the sill(s) of the variogram model.
            The default is max(distances).
        vg_nugget : scalar, optional
            Initial estimate of the nugget of the variogram model. In the case
            where vg_nugget is [], variogramfit fixes the nugget to 0 (default).
        vg_model : string or list of strings, optional
            Model type(s) of the variogram model. supported functions:
                - 'blinear'
                - 'circular'
                - 'cubic'
                - 'spherical'
                - 'pentashperical'
                - 'exponential'
                - 'gaussian'
                - 'whittle'
            The default is 'exponential'.
        weightfun : string, optional
            Type of weight function applied to the least squared error estimator:
                - 'none' least squared error
                - 'cressie85' weighted squared error
                              (nr_pairs/values_fun**2)
                - 'mcbratney86' weighted squared error
                               (nr_pairs*values_exp/values_fun**3)
                - '1/Yhat' weighted squared error (1/values_fun)
                - '1/Yhat2' weighted squared error (1/values_fun**2)
            The default is 'none'.
        plotit : boolean, optional
            Plot experimental and theoretical variogram.
            The default is False.
        full : boolean, optional
            Return only the optimized variogram model parameters (False) or
            return a full overview of the optimization (True).
            The default is False.

        Returns
        -------
        range : scalar or list
            Range value(s) of the (nested) variogram model, NOT the effective range.
        sill : scalar or list
            Sill value(s) of the (nested) variogram model.
        nugget : scalar
            Nugget of the (nested) variogram model.
        model : string or list of strings
            Model type of the (nested) variogram model.
        distances : array (1-by-k), optional
            The distances of the experimental variogram lag classes.
            Calculated according to 'variogram'.
        values_exp : array (1-by-k), optional
            Calculated experimental variogram values.
            Calculated according to 'variogram'.
        values_fun : array (1-by-k), optional
            Calculated variogram values according to the fitted model.
        nr_pairs : array (1-by-k), optional
            Number of point combinations grouped in the corresponding lag class.
            Calculated according to 'variogram'.
        weightfun : string, optional
            Weight function used to optimize the variogram fitting.
        residuals : array (1-by-k), optional
            Difference between the experimental and theoretical variogram values.
        weigths_residuals : array (1-by-k), optional
            Weights calculated according to the selected weight function.
        mse : scalar, optional
            Mean (weighted) squared error.

        """

        # Check compatibility of dimensions
        if not len(distances) == len(values):
            raise ValueError('The length of the values array has to match ' +
                             'the length of distances')

        if nr_pairs is not None and not len(distances) == len(nr_pairs):
            raise ValueError('The length of the nr_pairs array has to match ' +
                             'the length of distances')

        # Remove NaNs over all arrays
        if nr_pairs is not None:
            distances = np.asarray(distances).reshape(1, -1)
            values = np.asarray(values).reshape(1, -1)
            nr_pairs = np.asarray(nr_pairs).reshape(1, -1)

            exp_vg = np.vstack((distances, values, nr_pairs))
        else:
            distances = np.asarray(distances).reshape(1, -1)
            values = np.asarray(values).reshape(1, -1)

            exp_vg = np.vstack((distances, values))

        exp_vg = exp_vg[:, ~np.isnan(exp_vg).any(axis=0)]

        # convert model input to list if only one model type is inserted
        if isinstance(vg_model, str):
            vg_model = [vg_model]
        vg_model = [v.lower() for v in vg_model]

        # Calculate and store default parameter values (dependent on distance and
        # values arrays) as array
        if vg_range == []:
            vg_range = np.max(exp_vg[0, :]) * 2 / 3
        vg_range = np.ravel(vg_range)

        if vg_sill == []:
            vg_sill = np.max(values)
        vg_sill = np.ravel(vg_sill)

        nugget_defined = True
        if vg_nugget == []:
            vg_nugget = 0
            nugget_defined = False
        vg_nugget = np.ravel(vg_nugget)

        # Collect and expand the parameters in a 'parameter matrix' to give with
        # the optimization function
        nr_nested = np.max([len(vg_range), len(vg_sill),
                            len(vg_nugget), len(vg_model)])

        if len(vg_range) != nr_nested:
            vg_range = np.append(vg_range,
                                 np.repeat(vg_range[-1],
                                           nr_nested - len(vg_range)))

        if len(vg_sill) != nr_nested:
            vg_sill = np.append(vg_sill,
                                np.repeat(vg_sill[-1],
                                          nr_nested - len(vg_sill)))

        # Expand nugget too even though only the first value is considered
        if len(vg_nugget) != nr_nested:
            vg_nugget = np.append(vg_nugget,
                                  np.repeat(vg_nugget[-1],
                                            nr_nested - len(vg_nugget)))
        if len(vg_model) != nr_nested:
            vg_model.extend([vg_model[-1]] * (nr_nested - len(vg_model)))

        par_matrix = np.column_stack((vg_range, vg_sill, vg_nugget))

        # Remove the nugget during the fit optimization
        if not nugget_defined:
            par_matrix = np.delete(par_matrix, -1, axis=1)

        # Correct ranges for unbound models
        for i, i_model in enumerate(vg_model):
            if i_model == 'exponential':
                par_matrix[i, 0] = par_matrix[i, 0] / 3
            elif i == 'gaussian':
                par_matrix[i, 0] = par_matrix[i, 0] / 2

        # optimize.minimize can only handle arrays as inputs
        par_array = par_matrix.flatten()

        # Because the Nelder-Mead simplex method cannot handle boundaries,
        # the inputs are transformed to ensure positive values.
        par_array = np.abs(par_array)

        opt_fit = optimize.minimize(objectiveFun, par_array,
                                    args=(nugget_defined, vg_model,
                                          exp_vg, weightfun),
                                    method='Nelder-Mead')

        par_matrix = opt_fit.x.reshape(-1, 2 + nugget_defined)  # Recreate par_matrix
        par_matrix = np.abs(par_matrix)

        # Extract or define nugget
        if nugget_defined:
            n_out = par_matrix[0, 2]
        else:
            n_out = 0

        # Create plot
        if plotit:
            x_fun = np.linspace(0, np.max(distances), 1000).reshape(1, -1)
            plt.plot(distances, values, 'ko',
                     x_fun.T, variogramFun(x_fun, par_matrix[:, 0], par_matrix[:, 1],
                                           n_out, vg_model).T, 'b-')
            plt.xlabel('Lag distance (h)')
            plt.ylabel('Variogram value ($\gamma(h))$')
            plt.ylim(0)
            plt.title('Variogram')

            plt.show()

        # Output
        if not full:
            return par_matrix[:, 0], par_matrix[:, 1], n_out, vg_model
        else:
            # Calculate output variables of the 'full' version
            values_fun = variogramFun(distances, par_matrix[:, 0],
                                      par_matrix[:, 1], n_out, vg_model)
            residuals = values - values_fun
            weights_residuals = weightFun(values_fun, values, nr_pairs,
                                          weightfun)
            mse = (np.sum(residuals ** 2 * weights_residuals) /
                   np.sum(weights_residuals))

            Range = par_matrix[:, 0]
            sill = par_matrix[:, 1]
            nugget = n_out
            vg_model = vg_model
            mse = mse

            listvar.extend([Range] + [sill] + [nugget] + [vg_model] + [mse])


    def xmlfile():
        Range = float(listvar[0])
        sill = float(listvar[1])
        nugget = float(listvar[2])
        #vg_model = str(listvar[3])
        mse = float(listvar[4])
        contribution = float(sill-nugget)

        for i in listvar[3]:
            vg_model = str(i)
            vg_model = vg_model.title()

        line = '<parameters>  <algorithm name="kriging" />     <Variogram  nugget="' + str(nugget) + '" structures_count="1"  >    <structure_1  contribution="' + str(contribution) + '"  type="' + str(vg_model) + '"   >      <ranges max="' + str(Range) + '"  medium="' + str(Range) + '"  min="0"   />      <angles x="0"  y="0"  z="0"   />    </structure_1>  </Variogram>    <Grid_Name value="computation_grid" region=""  />     <Property_Name  value="test" />     <Kriging_Type  type="Ordinary Kriging (OK)" >    <parameters />  </Kriging_Type>    <Hard_Data  grid="sgems" region="" property="parameter"  />     <Min_Conditioning_Data  value="0" />     <Max_Conditioning_Data  value="12" />     <Search_Ellipsoid  value="' + str(Range) + ' ' + str(Range) + ' 0  0 0 0" />    <AdvancedSearch  use_advanced_search="0"></AdvancedSearch>  </parameters>'
        line = [line]
        np.savetxt(inputfile + 'variogram.xml', line, delimiter="\n", fmt="%s")

    def objectiveFun(params, nugget_defined, model, exp_vg, weightfun):
        """
        Handle for calculating the objective function that needs minimizing during
        the variogram fitting.

        Parameters
        ----------
        params : array
            Variogram parameter matrix where the first column represents the
            range, the second collumn represents the sill and the third
            (if present) the nugget. Every variogram component is represented by
            its own row, except for the nuggget effect, this is combined in the
            first row. Therefore, only the first nugget entry ([0,2]) is used for
            calculating the objective value.
        nugget_defined : boolean
            Logical value that defines wether a pure nugget effect is included.
            If the value is set to false, the nugget effect is neglected (fixed
            to 0).
        model : string or list of strings
            Model type of the (nested) variogram model.
        exp_vg : array
            Matrix containing the distances and the respective experimental
            variogram values (and number of pairs) as row vectors.
        weightfun : string
            Type of weight function applied to the least squared error estimator:
                - 'none' least squared error
                - 'cressie85' weighted squared error
                              (nr_pairs/values_fun**2)
                - 'mcbratney86' weighted squared error
                               (nr_pairs*values_exp/values_fun**3)
                - '1/Yhat' weighted squared error (1/values_fun)
                - '1/Yhat2' weighted squared error (1/values_fun**2)

        Returns
        -------
        values_obj : scalar
            Objective value (weighted squared error).

        """

        # Redefine shape of flattened matrix
        params = np.abs(params)
        params = params.reshape(-1, 2 + nugget_defined)

        if nugget_defined:
            nugget = params[0, -1]
        else:
            nugget = 0

        if exp_vg.shape[0] == 2:
            nr_pairs = None
        else:
            nr_pairs = exp_vg[2, :]

        values_fun = variogramFun(exp_vg[0, :],
                                  params[:, 0], params[:, 1], nugget, model)

        values_obj = np.sum(((values_fun - exp_vg[1, :]) ** 2)
                            * weightFun(values_fun, exp_vg[1, :],
                                        nr_pairs, weightfun))
        return values_obj


    def variogramFun(distances, vg_range, vg_sill, vg_nugget, vg_model, full=False):
        """
        Calculate the variogram values of the distance array for a variogram model
        represented by the variogram parameters range, sill, nugget and model type.

        Parameters
        ----------
        distances : list or array
            Distances for which the variogram values need to be calculated.
        vg_range : scalar, list or array
            Ranges of the variogram components.
        vg_sill : scalar, list or array
            Sills of the variogram components.
        vg_nugget : scalar
            Nugget of the variogram model.
        vg_model : string or list of strings
            Model type(s) of the variogram components. supported functions:
                - 'blinear'
                - 'circular'
                - 'cubic'
                - 'spherical'
                - 'pentashperical'
                - 'exponential'
                - 'gaussian'
                - 'whittle'
        full : boolean, optional
            Selector wether a full output is expected or the summarized version.
            The full version gives the variogram values of every variogram
            component, while the summarized output just returns the total value.
            The default is False (summarized output).

        Returns
        -------
        values : array
            Calculated variogram values for the respective distances.
        values_full : array
            Calculated variogram values of each model component for the
            respective distances.

        """

        # Converision of inputs to suitable formats
        h = np.ravel(distances)
        a = np.ravel(vg_range)
        c = np.ravel(vg_sill)
        n = np.ravel(vg_nugget)
        if isinstance(vg_model, str):
            vg_model = [vg_model]
        model = [v.lower() for v in vg_model]

        # Expand variogram parameter values to fit the nested structure
        nr_nested = np.max([len(a), len(c), len(model)])

        if len(a) != nr_nested:
            a = np.append(a, np.repeat(a[-1], nr_nested - len(a)))

        if len(c) != nr_nested:
            c = np.append(c, np.repeat(c[-1], nr_nested - len(c)))

        if len(vg_model) != nr_nested:
            vg_model.extend([vg_model[-1]] * (nr_nested - len(vg_model)))

        # Calculate variogram components
        if n[0] == 0:
            values_full = np.zeros((nr_nested, h.shape[0]))
        else:
            values_full = np.zeros((nr_nested + 1, h.shape[0]))
            values_full[-1, :] = n[0]

        for i in range(nr_nested):
            if model[i] == 'spherical':
                values_full[i, :] = ((h <= a[i]) * c[i] * ((3 * h / (2 * a[i]))
                                                           - 1 / 2 * (h / a[i]) ** 3)  # unbounded part
                                     + (h > a[i]) * c[i])  # bound part
            elif model[i] == 'pentaspherical':
                values_full[i, :] = ((h <= a[i]) * c[i] * (15 * h / (8 * a[i])
                                                           - 5 / 4 * (h / a[i]) ** 3 + 3 / 8 * (
                                                                       h / a[i]) ** 5)  # unbounded part
                                     + (h > a[i]) * c[i])  # bounded part
            elif model[i] == 'blinear':
                values_full[i, :] = ((h <= a[i]) * c[i] * (h / a[i])  # unbounded part
                                     + (h > a[i]) * c[i])  # bounded part
            elif model[i] == 'circular':
                values_full[i, :] = ((h <= a[i]) * c[i] * (1 - (2 / np.pi)
                                                           * np.acos(h / a[i]) + 2 * h / (np.pi * a[i])
                                                           * np.sqrt(1 - (h ** 2) / (a[i] ** 2)))  # unbounded part
                                     + (h > a[i]) * c[i])  # bounded part
            elif model[i] == 'cubic':
                values_full[i, :] = ((h <= a[i]) * c[i] * (7 * (h / a[i]) ** 2
                                                           - 35 / 4 * (h / a[i]) ** 3 + 7 / 2 * (h / a[i]) ** 5
                                                           - 3 / 4 * (h / a[i]) ** 7)  # unbounded part
                                     + (h > a[i]) * c[i])  # bounded part
            elif model[i] == 'exponential':
                values_full[i, :] = c[i] * (1 - np.exp(-h / a[i]))  # unbounded
            elif model[i] == 'gaussian':
                values_full[i, :] = c[i] * (1 - np.exp(-(h ** 2) / (a[i] ** 2)))  # unbounded
            elif model[i] == 'whittle':
                values_full[i, :] = c[i] * (1 - h / a[i] * kv(1, h / a[i]))  # unbounded
            # Need some extra work to add additional parameters to the fitting
            # elif model[i] == 'stable'
            #     values_full[i,:] = c[i]*(1
            #         - np.exp(-(h**param[i])/(a[i]**param[i]))) # unbounded
            # elif model[i] == 'matern'
            #     values_full[i,:] = c[i]*(1 - (1/((2**(param[i]-1))
            #         * gamma(param[i]))) * (h/a[i])**param[i]
            #         * kv(param[i], h/a[i])) # unbounded
            else:
                raise ValueError('model type unknown')

        # Calculate the full variogram model values
        values = np.sum(values_full, axis=0)

        if not full:
            return values.reshape((1, -1))
        return values.reshape((1, -1)), values_full


    def weightFun(values_fun, values_exp, nr_pairs=None, weightfun='none'):
        """
        Calculate the weight value given the fitted variogram values, the
        experimental variogram values, and the number of observations.

        Parameters
        ----------
        values_fun : array
            Variogram values calculated by the variogram model.
        values_exp : array
            Experimental variogram values.
        nr_pairs : array, optional
            Number of pairs per lag distances. Only necessary for calculating the
            'cressie85' or 'mcbratney86' weighted error.
            The default is None.
        weightfun : string, optional
            type of weight function applied to the squared error estimator:
                - 'none' least squared error
                - 'cressie85' weighted squared error
                              (nr_pairs/values_fun**2)
                - 'mcbratney86' weighted squared error
                               (nr_pairs*values_exp/values_fun**3)
                - '1/Yhat' weighted squared error (1/values_fun)
                - '1/Yhat2' weighted squared error (1/values_fun**2).
            The default is 'none'.

        Returns
        -------
        weights : array
            Weights corresponding to the respective residuals.

        """

        # Conversion inputs to suitable formats
        values_fun = np.ravel(values_fun)
        values_exp = np.ravel(values_exp)
        if nr_pairs is not None:
            nr_pairs = np.ravel(nr_pairs)
        weightfun = weightfun.lower()

        # Check compatibility of dimensions
        if not len(values_fun) == len(values_exp):
            raise ValueError('The length of the experimental values array has to ' +
                             'match the length of model values')

        if nr_pairs is not None and not len(values_fun) == len(nr_pairs):
            raise ValueError('The length of the nr_pairs array has to match ' +
                             'the length of model values')

        # Calculate weights
        if weightfun == 'none':
            weights = np.ones(values_fun.shape)
        elif weightfun == 'cressie85':
            weights = (nr_pairs / values_fun ** 2) / np.sum(nr_pairs / values_fun ** 2)
        elif weightfun == 'mcbratney86':
            weights = ((nr_pairs * values_exp / values_fun ** 3)
                       / np.sum(nr_pairs * values_exp / values_fun ** 3))
        elif weightfun == '1/yhat':
            weights = (1 / values_fun) / np.sum(1 / values_fun)
        elif weightfun == '1/yhat2':
            weights = 1 / values_fun ** 2 / np.sum(1 / values_fun ** 2)
        else:
            raise ValueError('weight function type unknown')

        return weights

    variogramfit(distances,values,nr_pairs,vg_range, vg_sill, vg_nugget)
    xmlfile()
    logger.info(f"ran variogram algorithm in {time.time() - start} s")


