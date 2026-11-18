Installation
============

Requirements
------------

The package metadata lists Python 3.6 or higher. Installation requires versions
of the following dependencies that support your Python version and platform.

* numpy >= 1.18, < 2.3.5
* scipy >= 1.5, < 2.0
* scikit-image >= 0.19, < 0.26
* scikit-learn >= 1.0, < 2.0
* opencv-python-headless >= 4.7.0, < 5.0.0

Install from PyPI
-----------------

The easiest way to install PySODMetrics is from PyPI using pip:

.. code-block:: bash

   pip install pysodmetrics

This is the **recommended and most stable** installation method.

Install from Source
-------------------

Installing from GitHub (Latest Version)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the latest development version (which may include new features but could be less stable):

.. code-block:: bash

   pip install git+https://github.com/lartpang/PySODMetrics.git

Installing from Cloned Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to modify the code or contribute to the project:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/lartpang/PySODMetrics.git
      cd PySODMetrics

2. Install in development mode:

   .. code-block:: bash

      pip install -e .

Building Documentation
----------------------

Install the package and its documentation dependencies from the repository root,
then build the HTML documentation.

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -b html deploy deploy/_build

Open ``deploy/_build/index.html`` to preview the documentation.

Verifying Installation
----------------------

To verify that PySODMetrics is installed correctly, open a Python interpreter and try:

.. code-block:: python

   import py_sod_metrics
   from py_sod_metrics import MAE, Smeasure

   # If no errors occur, the installation was successful!
   print("PySODMetrics installed successfully!")

You can also check the available classes:

.. code-block:: python

   import py_sod_metrics
   print(dir(py_sod_metrics))
