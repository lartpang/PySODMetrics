Supported Metrics
=================

This page provides detailed information about all the metrics supported by PySODMetrics.

Overview
--------

PySODMetrics provides two types of metric computation:

* **Sample-based**: Metrics are computed for each sample individually and then aggregated
* **Whole-based**: Metrics are computed across all samples globally

Most metrics support different aggregation strategies:

* ``max``: Maximum value across all thresholds
* ``avg``: Average value across all thresholds
* ``adp``: Adaptive threshold (2 × mean of predictions)
* ``bin``: Binary threshold (typically 0.5 or fixed threshold)
* ``si-*``: Size-invariant variants for handling multi-scale objects

Basic Metrics
-------------

MAE (Mean Absolute Error)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures the pixel-wise absolute difference between prediction and ground truth.

.. math::

   MAE = \frac{1}{W \times H} \sum_{x=1}^{W} \sum_{y=1}^{H} |P(x,y) - G(x,y)|

where :math:`P` is the prediction, :math:`G` is the ground truth, and :math:`W \times H` is the image size.

**Reference:**

Perazzi et al., "Saliency filters: Contrast based filtering for salient region detection", CVPR 2012

**Usage:**

.. code-block:: python

   from py_sod_metrics import MAE

   mae = MAE()
   mae.step(pred, gt)
   results = mae.get_results()
   print(f"MAE: {results['mae']:.4f}")

S-measure (Structure Measure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluates structural similarity between prediction and ground truth, considering both region-aware and object-aware components.

.. math::

   S_m = \alpha \cdot S_o + (1 - \alpha) \cdot S_r

where :math:`S_o` is the object-aware structural similarity and :math:`S_r` is the region-aware structural similarity.

**Reference:**

Fan et al., "Structure-measure: A new way to evaluate foreground maps", ICCV 2017

**Usage:**

.. code-block:: python

   from py_sod_metrics import Smeasure

   sm = Smeasure()
   sm.step(pred, gt)
   results = sm.get_results()
   print(f"S-measure: {results['sm']:.4f}")

E-measure (Enhanced-alignment Measure)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Captures both local and global matching information between prediction and ground truth.

**Reference:**

Fan et al., "Enhanced-alignment Measure for Binary Foreground Map Evaluation", IJCAI 2018

**Usage:**

.. code-block:: python

   from py_sod_metrics import Emeasure

   em = Emeasure()
   em.step(pred, gt)
   results = em.get_results()
   print(f"Max E-measure: {results['em']['adp']:.4f}")
   print(f"Avg E-measure: {results['em']['avg']:.4f}")

F-measure
~~~~~~~~~

Harmonic mean of precision and recall.

.. math::

   F_\beta = \frac{(1 + \beta^2) \times Precision \times Recall}{\beta^2 \times Precision + Recall}

**Reference:**

Achanta et al., "Frequency-tuned salient region detection", CVPR 2009

**Usage:**

.. code-block:: python

   from py_sod_metrics import Fmeasure

   fm = Fmeasure()
   fm.step(pred, gt)
   results = fm.get_results()
   print(f"Max F-measure: {results['fm']['adp']:.4f}")

Weighted F-measure
~~~~~~~~~~~~~~~~~~

A weighted version of F-measure that assigns different importance to different pixels based on their location.

**Reference:**

Margolin et al., "How to evaluate foreground maps?", CVPR 2014

**Usage:**

.. code-block:: python

   from py_sod_metrics import WeightedFmeasure

   wfm = WeightedFmeasure()
   wfm.step(pred, gt)
   results = wfm.get_results()
   print(f"Weighted F-measure: {results['wfm']:.4f}")

Advanced Metrics
----------------

FmeasureV2 Framework
~~~~~~~~~~~~~~~~~~~~

A flexible framework for computing multiple binary classification metrics using different handlers.

**Supported Handlers:**

* ``FmeasureHandler``: F-measure with configurable β
* ``PrecisionHandler``: Precision (Positive Predictive Value)
* ``RecallHandler``: Recall (Sensitivity, TPR)
* ``IOUHandler``: Intersection over Union
* ``DICEHandler``: Dice coefficient
* ``BERHandler``: Balanced Error Rate
* ``KappaHandler``: Cohen's Kappa
* ``OverallAccuracyHandler``: Overall classification accuracy
* ``SpecificityHandler``: Specificity (TNR)
* ``SensitivityHandler``: Sensitivity (same as Recall)
* ``FPRHandler``: False Positive Rate
* ``TNRHandler``: True Negative Rate
* ``TPRHandler``: True Positive Rate

**Usage:**

.. code-block:: python

   from py_sod_metrics import FmeasureV2, FmeasureHandler, IOUHandler

   fm_v2 = FmeasureV2(
       handlers={
           "fm": FmeasureHandler(beta=0.3),
           "iou": IOUHandler(),
       }
   )

   fm_v2.step(pred, gt)
   results = fm_v2.get_results()

Context-Measure
~~~~~~~~~~~~~~~

Designed specifically for camouflaged object detection, considering contextual information.

**Reference:**

Wang et al., "Context-measure: Contextualizing Metric for Camouflage", arXiv 2025

**Variants:**

* ``ContextMeasure``: Standard context measure :math:`C_\beta`
* ``CamouflageContextMeasure``: Weighted context measure :math:`C^\omega_\beta`

**Usage:**

.. code-block:: python

   from py_sod_metrics import ContextMeasure, CamouflageContextMeasure

   cm = ContextMeasure()
   ccm = CamouflageContextMeasure()

   cm.step(pred, gt)
   ccm.step(pred, gt)

Multi-Scale IoU (MSIoU)
~~~~~~~~~~~~~~~~~~~~~~~

Evaluates segmentation quality across multiple scales, particularly useful for fine structures.

**Reference:**

Ahmadzadeh et al., "Multiscale IOU: A Metric for Evaluation of Salient Object Detection with Fine Structures", ICIP 2021

**Usage:**

.. code-block:: python

   from py_sod_metrics import MSIoU

   msiou = MSIoU()
   msiou.step(pred, gt)
   results = msiou.get_results()

Human Correction Effort Measure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimates the effort required for humans to correct prediction errors.

**Reference:**

Qin et al., "Highly Accurate Dichotomous Image Segmentation", ECCV 2022

**Usage:**

.. code-block:: python

   from py_sod_metrics import HumanCorrectionEffortMeasure

   hcem = HumanCorrectionEffortMeasure()
   hcem.step(pred, gt)
   results = hcem.get_results()

Size-Invariant Metrics
----------------------

For datasets with objects at multiple scales, size-invariant variants provide more balanced evaluation.

Size-Invariant F-measure
~~~~~~~~~~~~~~~~~~~~~~~~~

**Reference:**

Li et al., "Size-invariance Matters: Rethinking Metrics and Losses for Imbalanced Multi-object Salient Object Detection", ICML 2024

**Usage:**

.. code-block:: python

   from py_sod_metrics import SizeInvarianceFmeasureV2

   si_fm = SizeInvarianceFmeasureV2()
   si_fm.step(pred, gt)
   results = si_fm.get_results()
   print(f"SI F-measure (avg): {results['fm']['si-avg']:.4f}")

Size-Invariant MAE
~~~~~~~~~~~~~~~~~~

**Usage:**

.. code-block:: python

   from py_sod_metrics import SizeInvarianceMAE

   si_mae = SizeInvarianceMAE()
   si_mae.step(pred, gt)
   results = si_mae.get_results()

Metric Comparison Table
-----------------------

+--------------------------------------------------+--------------------+------------------+
| Metric                                           | Sample-based       | Whole-based      |
+==================================================+====================+==================+
| MAE                                              | soft, si-soft      | —                |
+--------------------------------------------------+--------------------+------------------+
| S-measure                                        | soft               | —                |
+--------------------------------------------------+--------------------+------------------+
| Weighted F-measure                               | soft               | —                |
+--------------------------------------------------+--------------------+------------------+
| Human Correction Effort                          | soft               | —                |
+--------------------------------------------------+--------------------+------------------+
| Context-Measure                                  | soft               | —                |
+--------------------------------------------------+--------------------+------------------+
| Multi-Scale IoU                                  | max,avg,adp,bin    | —                |
+--------------------------------------------------+--------------------+------------------+
| E-measure                                        | max,avg,adp        | —                |
+--------------------------------------------------+--------------------+------------------+
| F-measure (V2)                                   | max,avg,adp,bin,si | bin,si           |
+--------------------------------------------------+--------------------+------------------+
| BER, Dice, IoU, Precision, Recall, etc.          | max,avg,adp,bin,si | bin,si           |
+--------------------------------------------------+--------------------+------------------+

Notes
-----

* **soft**: Metrics that work directly on continuous prediction values
* **si-**: Size-invariant variants that normalize by object size
* **adp**: Adaptive thresholding based on prediction statistics
* For detailed mathematical formulations, please refer to the original papers cited above
