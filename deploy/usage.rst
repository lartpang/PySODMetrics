Usage Guide
===========

The examples below evaluate paired segmentation predictions and ground-truth masks.

.. _metric-inputs:

Metric Inputs
-------------

Pass nonempty 2D NumPy arrays of the same shape to ``step(pred, gt)``. By default,
use grayscale ``uint8`` arrays in ``[0, 255]``. The library divides ``pred`` by
255 and min-max scales each nonconstant prediction to ``[0, 1]``. It converts the
ground truth to a boolean mask with ``gt > 128``.

Use ``step(pred, gt, normalize=False)`` to keep existing prediction values. This
mode requires ``float32`` or ``float64`` predictions in ``[0, 1]`` and a boolean
ground-truth mask. Adaptive and fixed binary thresholds use those values directly.
Dynamic curves still quantize predictions to 8 bits.
See :ref:`the threshold diagram <threshold-modes>` for the binary masks and
the order of the returned curves.

``CamouflageContextMeasure`` also takes ``img=img`` with an aligned RGB image of
shape ``(H, W, 3)``. Read color images from OpenCV as BGR and convert them to RGB
before passing them in.

Quick Start
-----------

Basic Example with Individual Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``pred_paths`` and ``gt_paths`` in matching sample order. This example uses
the legacy ``Fmeasure`` class. For new code, use :ref:`fmeasurev2`.

.. code-block:: python

   import cv2
   from py_sod_metrics import MAE, Emeasure, Smeasure, Fmeasure, WeightedFmeasure

   # Initialize metrics
   mae = MAE()
   em = Emeasure()
   sm = Smeasure()
   fm = Fmeasure()
   wfm = WeightedFmeasure()

   # Process your dataset
   # Note: pred and gt should be uint8 numpy arrays with values in [0, 255]
   for pred_path, gt_path in zip(pred_paths, gt_paths):
       pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
       gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

       # Resize prediction to match ground truth size if needed
       if pred.shape != gt.shape:
           pred = cv2.resize(pred, dsize=gt.shape[::-1], interpolation=cv2.INTER_LINEAR)

       # Feed predictions to metrics
       mae.step(pred, gt)
       em.step(pred, gt)
       sm.step(pred, gt)
       fm.step(pred, gt)
       wfm.step(pred, gt)

   # Get results
   mae_score = mae.get_results()["mae"]
   em_results = em.get_results()["em"]
   sm_score = sm.get_results()["sm"]
   fm_results = fm.get_results()["fm"]
   wfm_score = wfm.get_results()["wfm"]

   print(f"MAE: {mae_score:.4f}")
   print(f"S-measure: {sm_score:.4f}")
   print(f"Weighted F-measure: {wfm_score:.4f}")
   print(f"Max E-measure: {em_results['curve'].max():.4f}")
   print(f"Adaptive F-measure: {fm_results['adp']:.4f}")

Using FmeasureV2 Framework (Recommended)
-----------------------------------------

``FmeasureV2`` shares threshold statistics across the requested handlers. It returns
the enabled modes under each handler name. See :ref:`fmeasurev2` for the threshold
order and the difference between sample-based and pooled binary scores.

Basic FmeasureV2 Usage
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cv2
   from py_sod_metrics import FmeasureV2, FmeasureHandler, PrecisionHandler, RecallHandler, IOUHandler

   # Configure metric handlers
   fmv2 = FmeasureV2(
       metric_handlers={
           "fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
           "f1": FmeasureHandler(beta=1, with_adaptive=True, with_dynamic=True),
           "pre": PrecisionHandler(with_adaptive=True, with_dynamic=True),
           "rec": RecallHandler(with_adaptive=True, with_dynamic=True),
           "iou": IOUHandler(with_adaptive=True, with_dynamic=True),
       }
   )

   # Process dataset
   for pred_path, gt_path in zip(pred_paths, gt_paths):
       pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
       gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

       if pred.shape != gt.shape:
           pred = cv2.resize(pred, dsize=gt.shape[::-1], interpolation=cv2.INTER_LINEAR)

       fmv2.step(pred, gt)

   # Get results
   results = fmv2.get_results()

   # Access different aggregation strategies
   print(f"Adaptive F-measure: {results['fm']['adaptive']:.4f}")
   print(f"Mean F-measure: {results['fm']['dynamic'].mean():.4f}")
   print(f"Max F-measure: {results['fm']['dynamic'].max():.4f}")
   print(f"Adaptive Precision: {results['pre']['adaptive']:.4f}")
   print(f"Adaptive IoU: {results['iou']['adaptive']:.4f}")

**Available Handlers:**

- ``FmeasureHandler`` - F-measure with configurable :math:`\beta^2`
- ``PrecisionHandler`` - Precision (Positive Predictive Value)
- ``RecallHandler`` - Recall (Sensitivity, TPR)
- ``IOUHandler`` - Intersection over Union
- ``DICEHandler`` - Dice coefficient
- ``BERHandler`` - Balanced Error Rate
- ``KappaHandler`` - Cohen's Kappa
- ``OverallAccuracyHandler`` - Overall classification accuracy
- ``SpecificityHandler`` - Specificity (TNR)
- ``SensitivityHandler`` - Sensitivity (same as Recall)
- ``FPRHandler`` - False Positive Rate
- ``TNRHandler`` - True Negative Rate
- ``TPRHandler`` - True Positive Rate

Creating a Custom Metric Recorder
----------------------------------

For managing multiple metrics conveniently, you can create a custom recorder class.

Simple Metric Recorder
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from py_sod_metrics import MAE, Emeasure, Smeasure, Fmeasure, WeightedFmeasure, HumanCorrectionEffortMeasure

   class SimpleMetricRecorder:
       # A simple recorder for basic SOD metrics

       def __init__(self):
           self.mae = MAE()
           self.em = Emeasure()
           self.sm = Smeasure()
           self.fm = Fmeasure()
           self.wfm = WeightedFmeasure()
           self.hce = HumanCorrectionEffortMeasure()

       def step(self, pred, gt):
           # Update all metrics with a prediction-ground truth pair
           assert pred.shape == gt.shape
           assert pred.dtype == np.uint8 and gt.dtype == np.uint8

           self.mae.step(pred, gt)
           self.em.step(pred, gt)
           self.sm.step(pred, gt)
           self.fm.step(pred, gt)
           self.wfm.step(pred, gt)
           self.hce.step(pred, gt)

       def show(self, num_bits=3):
           # Get all metric results as a dictionary
           results = {}

           results['MAE'] = round(self.mae.get_results()['mae'], num_bits)
           results['Smeasure'] = round(self.sm.get_results()['sm'], num_bits)
           results['wFmeasure'] = round(self.wfm.get_results()['wfm'], num_bits)
           results['HCE'] = round(self.hce.get_results()['hce'], num_bits)

           em_results = self.em.get_results()['em']
           results['maxEm'] = round(em_results['curve'].max(), num_bits)
           results['avgEm'] = round(em_results['curve'].mean(), num_bits)
           results['adpEm'] = round(em_results['adp'], num_bits)

           fm_results = self.fm.get_results()['fm']
           results['maxFm'] = round(fm_results['curve'].max(), num_bits)
           results['avgFm'] = round(fm_results['curve'].mean(), num_bits)
           results['adpFm'] = round(fm_results['adp'], num_bits)

           return results

   # Usage example
   recorder = SimpleMetricRecorder()

   for pred, gt in dataset:
       recorder.step(pred, gt)

   results = recorder.show()
   print(results)

Advanced Metric Recorder with FmeasureV2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more comprehensive evaluation:

.. code-block:: python

   import numpy as np
   from py_sod_metrics import (
       MAE, Emeasure, Smeasure, WeightedFmeasure, HumanCorrectionEffortMeasure,
       FmeasureV2, FmeasureHandler, PrecisionHandler, RecallHandler,
       IOUHandler, DICEHandler, BERHandler, KappaHandler
   )

   class AdvancedMetricRecorder:
       # Advanced recorder supporting many metrics via FmeasureV2

       def __init__(self):
           # Individual metrics that don't use FmeasureV2
           self.mae = MAE()
           self.em = Emeasure()
           self.sm = Smeasure()
           self.wfm = WeightedFmeasure()
           self.hce = HumanCorrectionEffortMeasure()

           # FmeasureV2 with multiple handlers
           self.fmv2 = FmeasureV2(
               metric_handlers={
                   "fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
                   "f1": FmeasureHandler(beta=1, with_adaptive=True, with_dynamic=True),
                   "pre": PrecisionHandler(with_adaptive=True, with_dynamic=True),
                   "rec": RecallHandler(with_adaptive=True, with_dynamic=True),
                   "iou": IOUHandler(with_adaptive=True, with_dynamic=True),
                   "dice": DICEHandler(with_adaptive=True, with_dynamic=True),
                   "ber": BERHandler(with_adaptive=True, with_dynamic=True),
                   "kappa": KappaHandler(with_adaptive=True, with_dynamic=True),
               }
           )

       def step(self, pred, gt):
           # Update all metrics
           assert pred.shape == gt.shape
           assert pred.dtype == np.uint8 and gt.dtype == np.uint8

           self.mae.step(pred, gt)
           self.em.step(pred, gt)
           self.sm.step(pred, gt)
           self.wfm.step(pred, gt)
           self.hce.step(pred, gt)
           self.fmv2.step(pred, gt)

       def show(self, num_bits=3):
           # Get all results
           results = {}

           # Individual metrics
           results['MAE'] = round(self.mae.get_results()['mae'], num_bits)
           results['Smeasure'] = round(self.sm.get_results()['sm'], num_bits)
           results['wFmeasure'] = round(self.wfm.get_results()['wfm'], num_bits)
           results['HCE'] = round(self.hce.get_results()['hce'], num_bits)

           # E-measure
           em_data = self.em.get_results()['em']
           results['maxEm'] = round(em_data['curve'].max(), num_bits)
           results['avgEm'] = round(em_data['curve'].mean(), num_bits)
           results['adpEm'] = round(em_data['adp'], num_bits)

           # FmeasureV2 metrics
           fmv2_results = self.fmv2.get_results()
           for metric_name in ['fm', 'f1', 'pre', 'rec', 'iou', 'dice', 'ber', 'kappa']:
               metric_data = fmv2_results[metric_name]
               if 'dynamic' in metric_data:
                   results[f'max{metric_name}'] = round(metric_data['dynamic'].max(), num_bits)
                   results[f'avg{metric_name}'] = round(metric_data['dynamic'].mean(), num_bits)
               if 'adaptive' in metric_data:
                   results[f'adp{metric_name}'] = round(metric_data['adaptive'], num_bits)

           return results

   # Usage example
   recorder = AdvancedMetricRecorder()

   for pred, gt in dataset:
       recorder.step(pred, gt)

   results = recorder.show()
   for name, value in results.items():
       print(f"{name}: {value}")

Specialized Use Cases
---------------------

Context-Measure for Camouflaged Object Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provide ``cod_dataset`` as triples of prediction, mask, and aligned RGB image paths.

.. code-block:: python

   import cv2
   from py_sod_metrics import ContextMeasure, CamouflageContextMeasure

   # Standard Context Measure
   cm = ContextMeasure()

   # Camouflage Context Measure (weighted version, requires image)
   ccm = CamouflageContextMeasure()

   for pred_path, gt_path, img_path in cod_dataset:
       pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
       gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
       img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

       cm.step(pred, gt)
       ccm.step(pred, gt, img=img)

   cm_score = cm.get_results()['cm']
   ccm_score = ccm.get_results()['ccm']

   print(f"Context Measure: {cm_score:.4f}")
   print(f"Camouflage Context Measure: {ccm_score:.4f}")

Size-Invariant Metrics
~~~~~~~~~~~~~~~~~~~~~~

With the default ``sample_based=True``, each dynamic result is a list of per-image
target curves. The example computes the sample-based maximum in the order used by
the repository's example recorder. See :ref:`size-invariant-metrics` for the other
aggregation modes.

.. code-block:: python

   import numpy as np
   from py_sod_metrics import FmeasureHandler, SizeInvarianceFmeasureV2, SizeInvarianceMAE

   # Size-invariant MAE
   si_mae = SizeInvarianceMAE()

   # Size-invariant FmeasureV2
   si_fmv2 = SizeInvarianceFmeasureV2(
       metric_handlers={
           "fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
       }
   )

   # Process dataset
   for pred, gt in dataset:
       si_mae.step(pred, gt)
       si_fmv2.step(pred, gt)

   # Get results
   mae_score = si_mae.get_results()['si_mae']
   fm_scores = si_fmv2.get_results()["fm"]
   sample_max = np.mean([curves.max(axis=1).mean() for curves in fm_scores["dynamic"]])

   print(f"SI-MAE: {mae_score:.4f}")
   print(f"SI F-measure (sample-based max): {sample_max:.4f}")

Multi-Scale IoU
~~~~~~~~~~~~~~~

Choose one or more threshold modes. The result contains a key for each enabled mode.
``dynamic`` is the dataset-averaged 257-point curve over thresholds 0 to 256 in
ascending order. The evaluator compares the quantized 8-bit prediction with each
threshold using ``>=``. ``adaptive`` and ``binary`` are dataset-averaged scalars.
On the normalized prediction, adaptive mode uses ``pred >= min(2 * pred.mean(), 1)``
and binary mode uses ``pred > 0.5``.

.. code-block:: python

   from py_sod_metrics import MSIoU

   msiou = MSIoU(with_dynamic=True, with_adaptive=True, with_binary=True)

   for pred, gt in dataset:
       msiou.step(pred, gt)

   results = msiou.get_results()

   print(f"MS-IoU (adaptive): {results['adaptive']:.4f}")
   print(f"MS-IoU (max): {results['dynamic'].max():.4f}")
   print(f"MS-IoU (mean): {results['dynamic'].mean():.4f}")
   print(f"MS-IoU (binary): {results['binary']:.4f}")

Directory Evaluation Example
----------------------------

The following evaluator matches predictions and masks by filename. Each directory
contains one file per sample, with the same filename in both directories.

.. code-block:: python

   from pathlib import Path

   import cv2
   from py_sod_metrics import (
       MAE, Emeasure, Smeasure, WeightedFmeasure,
       FmeasureV2, FmeasureHandler, PrecisionHandler, RecallHandler, IOUHandler
   )

   class SODEvaluator:
       def __init__(self):
           self.mae = MAE()
           self.em = Emeasure()
           self.sm = Smeasure()
           self.wfm = WeightedFmeasure()

           self.fmv2 = FmeasureV2(
               metric_handlers={
                   "fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
                   "pre": PrecisionHandler(with_adaptive=True, with_dynamic=True),
                   "rec": RecallHandler(with_adaptive=True, with_dynamic=True),
                   "iou": IOUHandler(with_adaptive=True, with_dynamic=True),
               }
           )

       def step(self, pred, gt):
           self.mae.step(pred, gt)
           self.em.step(pred, gt)
           self.sm.step(pred, gt)
           self.wfm.step(pred, gt)
           self.fmv2.step(pred, gt)

       def get_results(self):
           results = {
               'MAE': self.mae.get_results()['mae'],
               'Smeasure': self.sm.get_results()['sm'],
               'wFmeasure': self.wfm.get_results()['wfm'],
           }

           em = self.em.get_results()['em']
           results.update({
               'maxEm': em['curve'].max(),
               'avgEm': em['curve'].mean(),
               'adpEm': em['adp'],
           })

           fmv2 = self.fmv2.get_results()
           for name in ['fm', 'pre', 'rec', 'iou']:
               data = fmv2[name]
               results[f'max{name}'] = data['dynamic'].max()
               results[f'avg{name}'] = data['dynamic'].mean()
               results[f'adp{name}'] = data['adaptive']

           return results

   def evaluate_predictions(pred_dir, gt_dir):
       evaluator = SODEvaluator()
       pred_dir = Path(pred_dir)
       gt_dir = Path(gt_dir)

       pred_names = {path.name for path in pred_dir.iterdir() if path.is_file()}
       gt_names = {path.name for path in gt_dir.iterdir() if path.is_file()}
       if not gt_names or pred_names != gt_names:
           raise ValueError("Expected matching prediction and mask filenames and at least one pair.")

       for name in sorted(gt_names):
           pred = cv2.imread(str(pred_dir / name), cv2.IMREAD_GRAYSCALE)
           gt = cv2.imread(str(gt_dir / name), cv2.IMREAD_GRAYSCALE)
           if pred is None or gt is None:
               raise ValueError(f"Could not read prediction or mask for {name}")

           if pred.shape != gt.shape:
               pred = cv2.resize(pred, dsize=gt.shape[::-1], interpolation=cv2.INTER_LINEAR)

           evaluator.step(pred, gt)

       results = evaluator.get_results()

       print("=" * 50)
       print("Evaluation Results")
       print("=" * 50)
       for metric, value in sorted(results.items()):
           print(f"{metric:20s}: {value:.4f}")

       return results

   # Run evaluation
   if __name__ == "__main__":
       pred_directory = "./predictions"
       gt_directory = "./ground_truth"
       results = evaluate_predictions(pred_directory, gt_directory)

Best Practices
--------------

1. **Data Format**

   - Use the formats in :ref:`metric-inputs`.
   - Resize predictions to the ground-truth dimensions before calling ``step()``.

2. **Memory Efficiency**

   - Use the ``step()`` method iteratively for large datasets
   - Call ``get_results()`` only once after processing all samples
   - Avoid loading all images into memory at once

3. **Result Interpretation**

   - Legacy F-measure and E-measure return ``adp`` and ``curve`` under ``fm`` and
     ``em``, respectively. Legacy F-measure returns precision and recall under ``pr``.
   - See :doc:`metrics` for threshold directions and aggregation. ``MSIoU`` uses
     257 thresholds, while the F-measure and E-measure curves use 256.

4. **Choosing Metrics**

   - **For SOD**: MAE, S-measure, E-measure, F-measure, Weighted F-measure
   - **For COD**: Add Context-Measure and Camouflage Context-Measure
   - **For images containing objects of different sizes**: Use size-invariant (SI) variants
   - **For fine structures**: Use Multi-Scale IoU
   - **For medical imaging**: Consider Dice coefficient and IoU

5. **Performance Tips**

   - Use FmeasureV2 to compute multiple related metrics efficiently
   - Specify only the metrics you need to save computation time

Reference
---------

For more examples, see the `examples folder <https://github.com/lartpang/PySODMetrics/tree/main/examples>`_ in the GitHub repository:

- ``metric_recorder.py`` - Metric recorder examples
- ``test_metrics.py`` - Regression tests against the bundled reference values
