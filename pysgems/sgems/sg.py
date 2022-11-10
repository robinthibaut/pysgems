#  Copyright (c) 2020. Robin Thibaut, Ghent University
import os
import subprocess
import time
import uuid
import warnings
from os.path import join as jp

from loguru import logger

from pysgems.utils.sgutils import joinlist


class Sgems:
    def __init__(
        self,
        project_name: str = "sgems_test",
        project_wd: str = "",
        res_dir: str = "",
        script_dir: str = "",
        exe_name: str = "",
        nodata: int = -9966699,  # sgems default value, do not change this
        check_env: bool = True,
        verbose: bool = True,
        parameters: list = [],
        kriging_type: str = "",
        algo_XML_list: list = [],
    ):
        """
        Initialize sgems object.

        :param project_name: Name of the project.
        :param project_wd: Project working directory.
        :param res_dir: Results directory.
        :param script_dir: Script directory.
        :param exe_name: Name of the sgems executable.
        :param nodata: No data value.
        :param check_env: Check if sgems is installed.
        :param verbose: Verbose mode.
        :param parameters: The column names from indicator data for Full Indicator Kriging.
        :param kriging_type: The type of kriging in case of FIK (=FIK).
        :param algo_XML_list: Paths to FIK XML file and post Kriging XML file.
        """

        self.verbose = verbose
        self.algo_XML_list = algo_XML_list
        self.parameters = parameters
        self.kriging_type = kriging_type

        if self.verbose:
            logger.add(jp(project_wd, f"{project_name}.log"), rotation="100 MB")
            logger.info(f"Project {project_name} initiated")

        if check_env:
            # First check if sgems installation files are in the user environment variables
            gstl_home = os.environ.get("GSTLAPPLIHOME")
            if not gstl_home:
                msg = "GSTLAPPLIHOME environment variable does not exist"
                warnings.warn(msg)
                if self.verbose:
                    logger.warning(msg)
            else:
                if self.verbose:
                    msg = "GSTLAPPLIHOME environment variable found"
                    logger.info(msg)
                path = os.getenv("Path")
                if gstl_home not in path:
                    msg = f"Variable {gstl_home} does not exist in Path environment variable"
                    warnings.warn(msg)
                    if self.verbose:
                        logger.warning(msg)
                if not exe_name:  # If no sgems exe file name is provided,
                    # checks for sgems exe file in the GSTLAPPLIHOME path
                    for file in os.listdir(gstl_home):
                        if (
                            file.endswith(".exe")
                            and ("sgems" in file)
                            and ("uninstall" not in file)
                        ):
                            exe_name = file
                if self.verbose:
                    msg = f"sgems exe file : {exe_name} in {gstl_home}"
                    logger.info(msg)

        # Project name
        self.project_name = project_name

        # Working directory
        self.project_wd = project_wd
        if not self.project_wd:
            self.project_wd = os.getcwd()

        # Results directory
        self.res_dir = res_dir
        # result directory generated according to project and algorithm name
        if self.res_dir is None:
            # Generate result directory if none is given
            self.res_dir = jp(
                self.project_wd,
                "results",
                "_".join([self.project_name, uuid.uuid1().hex]),
            )
        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        # Exe name
        self.exe_name = exe_name

        self.dis = None  # Discretization instance
        self.point_set = None  # Point set manager instance
        self.algo = None  # XML manipulation instance
        self.nodata = nodata

        self.object_file_names = []  # List of features name needed for the algorithm
        self.command_name = ""

        if not script_dir:
            dir_path = os.path.abspath(__file__ + "/../../")
            # Python template file path
            if kriging_type == "FIK":
                self.template_file = jp(
                    dir_path, "script_templates", "script_template_FIK.py"
                )
            else:
                self.template_file = jp(
                    dir_path, "script_templates", "script_template.py"
                )

    def write_command(self):
        """
        Write python script that sgems will run.
        """
        algo_XMLS = []
        self.command_name = jp(self.res_dir, f"{self.project_name}_commands.py")
        # This empty str will replace the # in front of the commands meant to execute sgems
        run_algo_flag = ""
        # within its python environment
        try:
            if self.kriging_type == "FIK":
                for i in range(len(self.algo_XML_list)):
                    if i == 0:
                        with open(self.algo.op_file) as alx:  # Remove unwanted \n
                            algo_xml = alx.read().strip("\n")
                    else:
                        algo_xml_name = f"{self.algo_XML_list[i]}.xml"
                        with open(algo_xml_name) as alx:  # Remove unwanted \n
                            algo_xml = alx.read().strip("\n")

                    algo_XMLS.append(algo_xml)

            else:
                with open(self.algo.op_file) as alx:  # Remove unwanted \n
                    algo_xml = alx.read().strip("\n")

            name = self.algo.root.find("algorithm").attrib["name"]  # Algorithm name
            try:
                # When performing simulations, sgems automatically add '__realn'
                # to the name of the nth generated property.
                nr = int(self.algo.root.find("Nb_Realizations").attrib["value"])
                name_op = "::".join([name + "__real" + str(i) for i in range(nr)])
            except AttributeError:
                name_op = name

        except AttributeError or FileNotFoundError:
            name = "None"
            name_op = name
            algo_xml = "None"
            run_algo_flag = "#"  # If no algorithm loaded, then just loads the data

        sgrid = [
            self.dis.ncol,
            self.dis.nrow,
            self.dis.nlay,
            self.dis.dx,
            self.dis.dy,
            self.dis.dz,
            self.dis.xo,
            self.dis.yo,
            self.dis.zo,
        ]  # Grid information
        grid = joinlist("::", sgrid)  # Grid in sgems format

        sgems_files = [f"{sf}.sgems" for sf in self.object_file_names]

        # The list below is the list of flags that will be replaced in the sgems python script
        # TODO: add option to change output file name (now default 'results.grid')
        if self.kriging_type == "FIK":
            params = [
                [run_algo_flag, "#~"],
                # for sgems convention...
                [self.res_dir.replace("\\", "//"), "RES_DIR"],
                [grid, "GRID"],
                [self.project_name, "PROJECT_NAME"],
                ["results", "FEATURE_OUTPUT"],  # results.grid = output file
                [name, "ALGORITHM_NAME"],
                ["", "OUTPUT_LIST"],
                [algo_XMLS[0], "ALGORITHM_XML1"],
                [algo_XMLS[1], "ALGORITHM_XML2"],
                [str(sgems_files), "OBJECT_FILES"],
                [str(self.parameters), "PARAMETERS"],
            ]
        else:
            params = [
                [run_algo_flag, "#~"],
                # for sgems convention...
                [self.res_dir.replace("\\", "//"), "RES_DIR"],
                [grid, "GRID"],
                [self.project_name, "PROJECT_NAME"],
                ["results", "FEATURE_OUTPUT"],  # results.grid = output file
                [name, "ALGORITHM_NAME"],
                [name_op, "OUTPUT_LIST"],
                [algo_xml, "ALGORITHM_XML"],
                [str(sgems_files), "OBJECT_FILES"],
            ]

        with open(self.template_file) as sst:
            template = sst.read()
        for i in range(len(params)):  # Replaces the parameters
            template = template.replace(params[i][1], params[i][0])

        with open(self.command_name, "w") as sstw:  # Write sgems python file
            sstw.write(template)

    def script_file(self):
        """Create script file"""
        run_script = jp(self.res_dir, "sgems.script")
        rscpt = open(run_script, "w")
        rscpt.write(" ".join(["RunScript", self.command_name]))
        rscpt.close()

    def bat_file(self):
        """Create bat file"""
        if not os.path.isfile(jp(self.res_dir, "sgems.script")):
            self.script_file()

        batch = jp(self.res_dir, "RunSgems.bat")
        bat = open(batch, "w")
        bat.write(" ".join(["cd", self.res_dir, "\n"]))
        bat.write(" ".join([self.exe_name, "sgems.script"]))
        bat.close()

    def run(self):
        """Call bat file, run sgems"""
        batch = jp(self.res_dir, "RunSgems.bat")
        if not os.path.isfile(batch):
            self.bat_file()
        start = time.time()

        try:
            os.remove(self.algo.op_file)
        except Exception:
            pass

        subprocess.call([batch])  # Opens the BAT file
        if self.verbose:
            logger.info(f"ran algorithm in {time.time() - start} s")
