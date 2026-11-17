Usage Guide
===========

This guide provides practical examples of how to use PySODMetrics for evaluating your image segmentation results.

Quick Start
-----------

Basic Example with Individual Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's a simple example using individual metrics:

.. code-block:: python

   import cv2
   import numpy as np
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

The ``FmeasureV2`` framework provides a unified interface for computing multiple metrics efficiently.

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

- ``FmeasureHandler`` - F-measure with configurable β
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

.. code-block:: python

   from py_sod_metrics import ContextMeasure, CamouflageContextMeasure

   # Standard Context Measure
   cm = ContextMeasure()

   # Camouflage Context Measure (weighted version, requires image)
   ccm = CamouflageContextMeasure()

   for pred_path, gt_path, img_path in cod_dataset:
       pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
       gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
       img = cv2.imread(img_path)  # RGB image

       cm.step(pred, gt)
       ccm.step(pred, gt, img)  # Note: CCM requires the original image

   cm_score = cm.get_results()['cm']
   ccm_score = ccm.get_results()['ccm']

   print(f"Context Measure: {cm_score:.4f}")
   print(f"Camouflage Context Measure: {ccm_score:.4f}")

Size-Invariant Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from py_sod_metrics import SizeInvarianceFmeasureV2, SizeInvarianceMAE, FmeasureHandler, PrecisionHandler, RecallHandler

   # Size-invariant MAE
   si_mae = SizeInvarianceMAE()

   # Size-invariant FmeasureV2
   si_fmv2 = SizeInvarianceFmeasureV2(
       metric_handlers={
           "si_fm": FmeasureHandler(beta=0.3, with_adaptive=True, with_dynamic=True),
           "si_pre": PrecisionHandler(with_adaptive=False, with_dynamic=True, sample_based=True),
           "si_rec": RecallHandler(with_adaptive=False, with_dynamic=True, sample_based=True),
       }
   )

   # Process dataset
   for pred, gt in dataset:
       si_mae.step(pred, gt)
       si_fmv2.step(pred, gt)

   # Get results
   mae_score = si_mae.get_results()['si_mae']
   fmv2_results = si_fmv2.get_results()

   print(f"SI-MAE: {mae_score:.4f}")

Multi-Scale IoU
~~~~~~~~~~~~~~~

.. code-block:: python

   from py_sod_metrics import MSIoU

   # Initialize with different strategies
   msiou = MSIoU(with_dynamic=True, with_adaptive=True, with_binary=True)

   for pred, gt in dataset:
       msiou.step(pred, gt)

   results = msiou.get_results()

   print(f"MS-IoU (adaptive): {results['adaptive']:.4f}")
   print(f"MS-IoU (max): {results['dynamic'].max():.4f}")
   print(f"MS-IoU (mean): {results['dynamic'].mean():.4f}")
   print(f"MS-IoU (binary): {results['binary']:.4f}")

Complete Evaluation Example
----------------------------

Here's a complete, production-ready example:

.. code-block:: python

   import os
   import cv2
   import numpy as np
   from py_sod_metrics import (
       MAE, Emeasure, Smeasure, WeightedFmeasure,
       FmeasureV2, FmeasureHandler, PrecisionHandler, RecallHandler, IOUHandler
   )

   class SODEvaluator:
       # Complete SOD evaluation class

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
       # Evaluate all predictions in a directory
       evaluator = SODEvaluator()

       pred_files = sorted(os.listdir(pred_dir))
       gt_files = sorted(os.listdir(gt_dir))

       assert len(pred_files) == len(gt_files), "Mismatch in number of files"

       for pred_file, gt_file in zip(pred_files, gt_files):
           pred = cv2.imread(os.path.join(pred_dir, pred_file), cv2.IMREAD_GRAYSCALE)
           gt = cv2.imread(os.path.join(gt_dir, gt_file), cv2.IMREAD_GRAYSCALE)

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

   - Predictions and ground truth should be ``uint8`` numpy arrays
   - Values should be in range [0, 255]
   - Ground truth masks should typically be binary (0 or 255)
   - Ensure prediction and ground truth have the same spatial dimensions

2. **Memory Efficiency**

   - Use the ``step()`` method iteratively for large datasets
   - Call ``get_results()`` only once after processing all samples
   - Avoid loading all images into memory at once

3. **Result Interpretation**

   - ``adaptive``: Threshold-based metric using 2× mean of predictions
   - ``dynamic``: Curve across all thresholds (256 points)
   - ``binary``: Metric computed on binarized predictions
   - ``curve``: Full precision-recall curve or threshold-based curve

4. **Choosing Metrics**

   - **For SOD**: MAE, S-measure, E-measure, F-measure, Weighted F-measure
   - **For COD**: Add Context-Measure and Camouflage Context-Measure
   - **For multi-scale objects**: Use size-invariant (SI) variants
   - **For fine structures**: Use Multi-Scale IoU
   - **For medical imaging**: Consider Dice coefficient and IoU

5. **Performance Tips**

   - Resize predictions to match ground truth size before calling ``step()``
   - Use FmeasureV2 to compute multiple related metrics efficiently
   - Specify only the metrics you need to save computation time

Reference
---------

For more examples, see the `examples folder <https://github.com/lartpang/PySODMetrics/tree/main/examples>`_ in the GitHub repository:

- ``metric_recorder.py`` - Production-ready metric recorder implementations
- ``test_metrics.py`` - Comprehensive test cases showing all features
