Installation
============

Requirements
------------

PySODMetrics requires Python 3.6 or higher and the following dependencies:

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

To build the documentation locally, you need to install the documentation dependencies:

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme

Then build the documentation:

.. code-block:: bash

   cd docs
   make html  # On Linux/Mac
   # or
   make.bat html  # On Windows

The built documentation will be available in ``docs/_build/html/``.

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
