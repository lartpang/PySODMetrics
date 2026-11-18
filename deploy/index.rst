PySODMetrics Documentation
===========================

Welcome to PySODMetrics - A simple and efficient implementation of SOD metrics.

.. image:: https://img.shields.io/pypi/v/pysodmetrics
   :target: https://pypi.org/project/pysodmetrics/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/dm/pysodmetrics?label=pypi%20downloads&logo=PyPI&logoColor=white
   :target: https://pypi.org/project/pysodmetrics/
   :alt: Downloads

Overview
--------

PySODMetrics is a Python library that provides simple and efficient implementations of
metrics for evaluating salient object detection (SOD), camouflaged object detection (COD),
and medical image segmentation tasks.

**Key Features:**

* Based on numpy and scipy for fast computation
* Core SOD metrics compared with Fan's MATLAB toolbox
* Simple and extensible code structure
* Lightweight and easy to use

The 1.7.0 regression results were checked on the maintainer's original Windows
machine. :ref:`MSIoU <msiou>` uses updated baselines after the edge-counting fix.
The other tested metrics matched the historical PySODMetrics baselines. See the `changelog
<https://github.com/lartpang/PySODMetrics/blob/main/CHANGELOG.md>`_.

.. note::
   Our exploration in this field continues with `PyIRSTDMetrics <https://github.com/lartpang/PyIRSTDMetrics>`_,
   a project born from the same core motivation. Think of them as twin initiatives:
   this project maps the landscape of current evaluation, while its sibling takes the next step
   to expand upon and rethink it.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   usage
   metrics

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

Supported Metrics
-----------------

PySODMetrics supports a comprehensive set of evaluation metrics:

* **MAE** - Mean Absolute Error
* **S-measure** (:math:`S_m`) - Structure Measure
* **E-measure** (:math:`E_m`) - Enhanced-alignment Measure
* **F-measure** (:math:`F_\beta`) - Precision-Recall F-measure
* **Weighted F-measure** (:math:`F^\omega_\beta`)
* **Context-Measure** (:math:`C_\beta`, :math:`C^\omega_\beta`)
* **Multi-Scale IoU** - Multi-scale Intersection over Union
* **Human Correction Effort Measure**
* **FmeasureV2 handlers** for IoU, Dice, BER, Kappa, precision, recall and other classification metrics
* **Size-invariant metrics** for MAE and the FmeasureV2 handlers

See :doc:`metrics` for all metric definitions. The illustrated examples cover
:ref:`threshold modes <threshold-modes>`, :ref:`MSIoU <msiou>`, and
:ref:`size-invariant metrics <size-invariant-metrics>`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Related Projects
================

* `PySODEvalToolkit <https://github.com/lartpang/PySODEvalToolkit>`_ - A Python-based Evaluation Toolbox for Salient Object Detection and Camouflaged Object Detection

Links
=====

* **GitHub Repository:** https://github.com/lartpang/PySODMetrics
* **PyPI Package:** https://pypi.org/project/pysodmetrics/
* **Issue Tracker:** https://github.com/lartpang/PySODMetrics/issues
