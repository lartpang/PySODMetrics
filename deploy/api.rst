API Reference
=============

See :ref:`metric-inputs` for ``step`` input formats and :doc:`metrics` for how to
interpret the scores. Call ``get_results()`` after accumulating at least one sample.

.. contents:: Contents
   :local:
   :depth: 2

Core Metrics Module
-------------------

.. automodule:: py_sod_metrics.sod_metrics
   :members:
   :undoc-members:
   :show-inheritance:

FmeasureV2 Module
-----------------

.. automodule:: py_sod_metrics.fmeasurev2
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: FmeasureV2, KappaHandler

.. autoclass:: py_sod_metrics.fmeasurev2.FmeasureV2
   :members:
   :exclude-members: dynamically_binarizing

   .. py:method:: dynamically_binarizing(pred, gt, FG, BG)

      Return a dictionary with ``tp``, ``fp``, ``tn``, and ``fn`` arrays, each with
      256 entries in threshold order 255 down to 0. The method compares the 8-bit
      quantized normalized prediction with each threshold using ``>=``. ``gt`` is
      a boolean mask, and ``FG`` and ``BG`` are its foreground and background counts.

.. py:class:: KappaHandler(with_dynamic, with_adaptive, *, with_binary=False, sample_based=True)

   Computes Cohen's kappa. :math:`\mathrm{OA}` is observed overall accuracy and
   :math:`p_e` is expected agreement by chance.

   .. math::

      \kappa = \frac{\mathrm{OA} - p_e}{1 - p_e}, \qquad
      p_e = \frac{(TP + FP)(TP + FN) + (TN + FN)(TN + TP)}{(TP + FP + TN + FN)^2}

   .. py:method:: __call__(tp, fp, tn, fn)

      Compute kappa from confusion counts. A zero denominator yields zero.

Context Measure Module
----------------------

.. automodule:: py_sod_metrics.context_measure
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Scale IoU Module
----------------------

.. automodule:: py_sod_metrics.multiscale_iou
   :no-members:

.. py:class:: MSIoU(with_dynamic, with_adaptive, *, with_binary=False, num_levels=10)

   Evaluates predicted edge coverage across grid scales. See :ref:`msiou` for the
   definition and scale configuration.

   .. py:method:: step(pred, gt, normalize=True)

      Accumulate scores for a prediction and ground-truth pair. Returns ``None``.

   .. py:method:: get_results()

      Return the mean over samples for each enabled mode. ``dynamic`` is a 256-point
      NumPy array in ascending threshold order from 0 to 255. ``adaptive`` and
      ``binary`` are scalars.

   .. py:method:: get_edge(mask)

      Return a ``float64`` edge map of zeros and ones using the Sobel operator.

   .. py:method:: shrink_by_grid(image, cell_size)

      Reduce a binary edge map to one occupancy value per grid cell. The result
      uses the input array's dtype. A positive cell size is required.

   .. py:method:: multi_scale_iou(pred_edge, gt_edge, gt_shrunk_cache=None)

      Return the edge coverage ratios from the finest to the coarsest grid.

   .. py:method:: binarizing(pred_bin, gt_edge, gt_shrunk_cache=None)

      Return a scalar edge score for a binary prediction and a ground-truth edge
      map. The method integrates the per-scale scores or returns the original-resolution
      score when using one level. The optional cache contains the ground-truth edge
      maps reduced to each grid scale.

Size Invariance Module
-----------------------

.. automodule:: py_sod_metrics.size_invariance
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: SizeInvarianceMAE, parse_connected_components

.. py:class:: SizeInvarianceMAE()

   Computes mean absolute error over target boxes and the remaining background.

   .. py:method:: step(pred, gt, normalize=True)

      Accumulate the size-invariant MAE for a prediction and ground-truth pair.

   .. py:method:: get_results()

      Return ``{"si_mae": score}`` with the mean over samples.

.. py:function:: parse_connected_components(mask, area_threshold=50)

   Return ``(num_targets, labels)`` for the 4-connected components of a binary
   mask. Select components at or above ``area_threshold``. If none meet the
   threshold, select all components tied for largest. Labels start at 1,
   and 0 marks pixels outside the selected components. An all-background mask
   returns zero targets and an all-zero label image.

Utility Functions
-----------------

.. automodule:: py_sod_metrics.utils
   :members:
   :undoc-members:
   :show-inheritance:
