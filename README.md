# pysgems 
Use SGeMS (Standford Geostatistical Modeling Software) within Python.

Contributors / feedback from users are welcome.

SGeMS home page: http://sgems.sourceforge.net/

The kriging example described in this file can be found in the 'demo.py' file.

## Installation

```bash
pip install pysgems
```

Users need to add this variable to their system environment variables:

Name: GSTLAPPLIHOME

Value: Path to your SGEMS folder (e.g. C:\Program Files (x86)\SGeMS)

It is also necessary to add the path of your SGEMS folder to the system variable 'Path'.

## Introduction

This package works by calling SGEMS via the command in the Command Prompt:
```bash
sgems sgems.script
```

The sgems.script contains a command that SGEMS will execute internally:

```bash
RunScript python_script.py
```

The python_script file contains a Python script (ver. <= 2.7.x) that SGEMS will execute, as if ran into the Run Script... window of SGEMS.

This package revolves around modifying this Python script file.

## Usage

This package was designed with 2D estimation/simulation problems in mind, it should support 3D grids but it hasn't been tested yet.

### Load data and generate computation grid

The package expects the classical ASCII GEOEAS data format.

```python
import os
from os.path import join as join_path

from pysgems import toolbox

# Define working directory
cwd = os.getcwd()
# Define datasets directory
data_dir = join_path(cwd, 'datasets', 'demo')
# Define file name
f_name = 'sgems_dataset.dat'
```

There are several ways to define grids in this package framework, the user can refer to the methods documentation.

The simplest way is to load the data and indicate cell dimensions (dx, dy, dz is 0 by default). If no spatial limits are imposed, they will be automatically generated to englobe all data points.

It is possible to take into accounts 'no-data points' by indicating the 'no-data' value.

```python
sgems = toolbox.Sgems(dx=2, dy=2, data_dir=data_dir, file_name=f_name, nodata=-999)
```

Optionally provide a results directory name, if none is provided, one will automatically be generated with an unique name.

```python
sgems.res_dir = join_path(cwd, 'results', 'demo')
```

The loaded dataset can be explored by:

```python
sgems.columns  # Display features name
sgems.dataframe  # Pandas DataFrame
```

### Get algorithm XML from SGeMS
Users must first run their algorithm inside SGeMS, view the Commands Panel, copy the algorithm XML and paste it into a XML file (e.g. algorithm.xml), and save this file into the 'algorithms' folder.

The algorithm XML starts and ends with:
```xml
<parameters> ... </parameters>
```

#### Edit algorithm

Let's take for example the kriging algorithm.

Read the 'kriging.xml' file in the following way:

```python
sgems.xml_reader('kriging')

```

Display the XML structure:

```python

sgems.show_tree()

```

It will output:

```python
Path
{attribute1: value1, ..., attribute_n: value_n}
```

For the kriging algorithm the first lines will look like this:

```python
algorithm
{'name': 'kriging_mean'}
Variogram
{'nugget': '0', 'structures_count': '1'}
Variogram//structure_1
{'contribution': '1', 'type': 'Spherical'}
Variogram//structure_1//ranges
{'max': '150', 'medium': '150', 'min': '150'}
```

To modify an element:

```python
sgems.xml_update(path, attribute_name, new_value)
```

For example:

```python
sgems.xml_update('Variogram', 'nugget', '0.01')
sgems.xml_update('Variogram', 'structure_1', 'type', 'Exponential')
```

The following is also acceptable:

```python
sgems.xml_update('Variogram//structure_1//ranges', new_attribute_dict={'max': '150', 'medium': '150', 'min': '150'})
```

A Python script template is stored in the main folder. To load the updated XML algorithm into the SGEMS Python script template:

```python
sgems.write_commands()
```

The files will be loaded in the results folder (automatically generated or user-defined).


### Exports needed point sets in SGEMS binary format.

In order to use the dataset within SGEMS, it is necessary to export the needed point sets in the SGEMS binary format.

Get the list of features name available to export:

```python
print(sgems.columns)
```

The line

```python
sgems.make_data('feature_name')
```

Will export the desired feature point set in SGEMS binary format to the results folder. Rows containing the no-data value will not be exported.

At this time, the make_data() method only supports point sets for individual features. It should be extended to support grids and multiple features.

### Run SGEMS

Run the algorithm via:

```python
sgems.run()
```

It will generate a batch file and a script file. The batch file can later be used outside of Python.

### Other features

It is possible to fix hard data to a grid given the hard data point set. See get_nodes() method.

## Contribution and points to improve

Contributors and feedback from users are welcome. 

The package features should be separated in proper classes in proper folders.

The package should be made more robust and able to support all SGEMS features.

The make_data() method should be extended to support grids and multiple features.

The package should be properly documented.

The package should have more built-in visualization methods.

The Python script template should be more generic and its modification more flexible.

More examples should be uploaded.
