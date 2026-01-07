import math

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .utils import EPS, TYPE, validate_and_normalize_input


class ContextMeasure:
    """Context-measure for evaluating foreground segmentation quality.

    This metric evaluates predictions by considering both forward inference (how well predictions align with ground truth) and reverse deduction (how completely ground truth is covered by predictions), using context-aware Gaussian kernels.

    ```
    @article{ContextMeasure,
        title={Context-measure: Contextualizing Metric for Camouflage},
        author={Wang, Chen-Yang and Ji, Gepeng and Shao, Song and Cheng, Ming-Ming and Fan, Deng-Ping},
        journal={arXiv preprint arXiv:2512.07076},
        year={2025}
    }
    ```
    """

    def __init__(self, beta2: float = 1.0, alpha: float = 6.0):
        """Initialize the Context Measure evaluator.

        Args:
            beta2 (float): Balancing factor between forward inference and reverse deduction. Higher values give more weight to forward inference. Defaults to 1.0.
            alpha (float): Scaling factor for Gaussian kernel covariance, controls the spatial context range. Defaults to 6.0.
        """
        self.beta2 = beta2
        self.alpha = alpha
        self._exp_factor = math.e / (math.e - 1)
        self.scores = []

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred and gt.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        # align with the original implementation
        pred = pred.astype(TYPE)
        gt = gt.astype(TYPE)

        score = self.compute(pred, gt, cd=np.zeros_like(gt))
        self.scores.append(score)

    def compute(self, pred: np.ndarray, gt: np.ndarray, cd: np.ndarray) -> float:
        """Compute the context measure between prediction and ground truth.

        Args:
            pred (np.ndarray): Prediction map (values between 0 and 1).
            gt (np.ndarray): Ground truth map (boolean or 0/1 values).
            cd (np.ndarray): Camouflage degree map (values between 0 and 1).

        Returns:
            float: Context measure value.
        """
        cov_matrix, x_dis, y_dis = self._compute_y_params(gt)
        K = self._gaussian_kernel(x_dis, y_dis, cov_matrix)

        # Forward inference: measure prediction relevance
        forward = self._forward_inference(pred, gt, K)
        mforward = np.sum(forward * pred) / (np.sum(pred) + EPS)

        # Reverse deduction: measure ground truth completeness
        reverse = self._reverse_deduction(pred, gt, K)
        wreverse = np.sum(reverse * (gt + cd)) / (np.sum(gt) + np.sum(cd) + EPS)

        # F-measure style combination
        return (1 + self.beta2) * mforward * wreverse / (self.beta2 * mforward + wreverse + EPS)

    def _forward_inference(self, X: np.ndarray, Y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Calculate forward inference: how well predictions align with ground truth context."""
        x_binary = (X > 0).astype(int)
        # note: using EPS=1e-8 and this statement, the test result is the same as the original implementation
        # global_relevance_matrix = cv2.filter2D(Y, cv2.CV_32F, kernel)
        # note: this is a hack to make sure that the type of Y is compatible with more diverse data
        global_relevance_matrix = cv2.filter2D(Y.astype(np.float32), cv2.CV_32F, kernel)
        return x_binary * global_relevance_matrix

    def _reverse_deduction(self, X: np.ndarray, Y: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Calculate reverse deduction: how completely ground truth is covered by predictions."""
        X = X.astype(float)
        non_global_completeness_matrix = np.exp(-1 * cv2.filter2D(X, -1, kernel))
        global_completeness_matrix = 1 - non_global_completeness_matrix
        reverse = self._exp_factor * Y * global_completeness_matrix
        return reverse

    def _gaussian_kernel(self, x_dis: int, y_dis: int, cov_matrix: np.ndarray) -> np.ndarray:
        """Generate a 2D Gaussian kernel based on covariance matrix."""
        det_sigma = np.linalg.det(cov_matrix)
        inv_sigma = np.linalg.inv(cov_matrix)

        x, y = np.meshgrid(np.arange(-x_dis, x_dis + 1), np.arange(-y_dis, y_dis + 1), indexing="ij")
        Z = np.stack([x, y], axis=-1)
        exp_term = np.einsum("...i,ij,...j->...", Z, inv_sigma, Z)

        kernel = np.exp(-0.5 * exp_term) / (2 * np.pi * np.sqrt(det_sigma))
        return kernel / np.sum(kernel)

    def _compute_y_params(self, Y: np.ndarray) -> tuple:
        """Compute Gaussian kernel parameters based on ground truth distribution."""
        points = np.argwhere(Y > 0)
        if len(points) <= 1:
            return np.diag([0.25, 0.25]), 1, 1

        cov_matrix = np.cov(points, rowvar=False)
        sigma_x = np.sqrt(cov_matrix[0, 0])
        sigma_y = np.sqrt(cov_matrix[1, 1])
        total_sigma = np.sqrt(cov_matrix[0, 0] + cov_matrix[1, 1])

        std_cov_matrix = self.alpha**2 * cov_matrix / (total_sigma**2)
        std_sigma_x = self.alpha * sigma_x / total_sigma
        std_sigma_y = self.alpha * sigma_y / total_sigma
        x_dis = round(3 * std_sigma_x)
        y_dis = round(3 * std_sigma_y)

        return std_cov_matrix, x_dis, y_dis

    def get_results(self) -> dict:
        """Return the results about context measure.

        Returns:
            dict(cm=context_measure)
        """
        cm = np.mean(np.array(self.scores, dtype=TYPE))
        return dict(cm=cm)


class CamouflageContextMeasure(ContextMeasure):
    """Camouflage Context-measure for evaluating camouflaged object detection quality.

    This metric extends the base ContextMeasure by incorporating camouflage degree, which measures how well the foreground blends with its surrounding background. It uses patch-based nearest neighbor matching in Lab color space with spatial constraints to estimate camouflage difficulty.

    ```
    @article{ContextMeasure,
        title={Context-measure: Contextualizing Metric for Camouflage},
        author={Wang, Chen-Yang and Ji, Gepeng and Shao, Song and Cheng, Ming-Ming and Fan, Deng-Ping},
        journal={arXiv preprint arXiv:2512.07076},
        year={2025}
    }
    ```
    """

    def __init__(self, beta2: float = 1.2, alpha: float = 6.0, gamma: int = 8, lambda_spatial: float = 20):
        """Initialize the Camouflage Context Measure evaluator.

        Args:
            beta2 (float): Balancing factor for forward and reverse. Defaults to 1.2 for camouflage.
            alpha (float): Gaussian kernel scaling factor. Defaults to 6.0.
            gamma (int): Exponential scaling factor for camouflage degree. Defaults to 8.
            lambda_spatial (float): Weight for spatial distance in ANN search. Defaults to 20.
        """
        super().__init__(beta2=beta2, alpha=alpha)
        self.gamma = gamma
        self.lambda_spatial = lambda_spatial

    def step(self, pred: np.ndarray, gt: np.ndarray, img: np.ndarray, normalize: bool = True):
        """Statistics the metric for the pair of pred, gt, and img.

        Args:
            pred (np.ndarray): Prediction, gray scale image.
            gt (np.ndarray): Ground truth, gray scale image.
            img (np.ndarray): Original RGB image (required for camouflage degree calculation).
            normalize (bool, optional): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        pred = pred.astype(TYPE)
        gt = gt.astype(TYPE)

        _, cd = self._calculate_camouflage_degree(img, gt)
        score = self.compute(pred, gt, cd=cd)
        self.scores.append(score)

    def _calculate_camouflage_degree(self, img: np.ndarray, mask: np.ndarray, w: int = 7) -> tuple:
        """Compute the camouflage degree matrix using Lab+spatial ANN and RGB reconstruction.

        Args:
            img (np.ndarray): RGB image (H x W x 3).
            mask (np.ndarray): Binary mask (H x W).
            w (int): Patch size. Defaults to 7.

        Returns:
            tuple: (reconstructed_image, camouflage_degree_matrix)
        """
        mask_binary = (mask > 0).astype(np.uint8)
        fg_mask = mask_binary
        bg_mask = self._extract_surrounding_background(fg_mask, kernel_size=20)
        im_fg = fg_mask[:, :, np.newaxis] * img
        im_bg = bg_mask[:, :, np.newaxis] * img
        im_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # Step 1: Extract patches in Lab space
        im_fg_lab = im_lab * fg_mask[:, :, np.newaxis]
        im_bg_lab = im_lab * bg_mask[:, :, np.newaxis]

        fg_indices, fg_feat_lab = self._extract_patches(im_fg_lab, fg_mask, w, d=w // 2)
        bg_indices, bg_feat_lab = self._extract_patches(im_bg_lab, bg_mask, w, d=w // 2)

        # Check if we have enough patches to compute camouflage degree
        if len(fg_indices) == 0 or len(bg_indices) == 0:
            # Return zero camouflage degree when insufficient data
            img_recon = np.zeros_like(img)
            cd = np.zeros_like(mask, dtype=TYPE)
            return img_recon, cd

        # Step 2: Lab+spatial ANN query
        fg_nn = self._ann_with_spatial_faiss(bg_feat_lab, fg_feat_lab, bg_indices, fg_indices)

        # Step 3: Reconstruct foreground in RGB space
        img_recon = self._reconstruct_image(img, fg_indices, bg_indices, fg_nn, im_bg, w)

        # Step 4: Compute similarity in Lab space
        similarity_matrix = self._compute_delta_e2000_matrix(img_recon, im_fg.astype(np.uint8)).astype(TYPE)

        # Step 5: Compute camouflage degree
        cd = ((np.exp(self.gamma * similarity_matrix * mask_binary) - 1) / (np.exp(self.gamma) - 1)).astype(TYPE)

        return img_recon, cd

    def _ann_with_spatial_faiss(self, x, q, x_coords, q_coords, m=16):
        """Approximate Nearest Neighbor search with spatial constraints using sklearn.

        Note: Method name retained for compatibility, but now uses sklearn.neighbors.NearestNeighbors instead of FAISS for a more lightweight dependency.
        """
        all_coords = np.vstack([x_coords, q_coords])
        scaled_coords = StandardScaler().fit_transform(all_coords)
        x_coords_scaled = scaled_coords[: len(x_coords)]
        q_coords_scaled = scaled_coords[len(x_coords) :]

        x_aug = np.hstack([x, self.lambda_spatial * x_coords_scaled]).astype(np.float32)
        q_aug = np.hstack([q, self.lambda_spatial * q_coords_scaled]).astype(np.float32)

        # Use sklearn NearestNeighbors instead of FAISS for lightweight alternative
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
        nbrs.fit(x_aug)

        _, indices = nbrs.kneighbors(q_aug)  # top-1
        return indices

    def _extract_surrounding_background(self, mask: np.ndarray, kernel_size: int = 20) -> np.ndarray:
        """Extract the surrounding background region around the foreground."""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        surrounding_bg_mask = dilated_mask - mask
        return surrounding_bg_mask

    def _extract_patches(self, img: np.ndarray, mask: np.ndarray, w: int, d: int) -> tuple:
        """Extract valid patches from the image based on mask."""
        h, w_, c = img.shape
        pad_h = (d - (h - w) % d) % d
        pad_w = (d - (w_ - w) % d) % d
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")

        new_h, new_w = img_padded.shape[:2]

        img_patches = np.lib.stride_tricks.sliding_window_view(img_padded, (w, w, img.shape[2]))[::d, ::d, 0, :, :, :]
        mask_patches = np.lib.stride_tricks.sliding_window_view(mask_padded, (w, w))[::d, ::d, :, :]

        img_patches = img_patches.reshape(-1, w * w * c)
        mask_patches = mask_patches.reshape(-1, w, w)

        grid_x, grid_y = np.meshgrid(np.arange(0, new_h - w + 1, d), np.arange(0, new_w - w + 1, d), indexing="ij")
        all_indices = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        valid_idx = np.all(mask_patches > 0, axis=(1, 2))
        valid_indices = all_indices[valid_idx]
        valid_patches = img_patches[valid_idx]

        return valid_indices, valid_patches

    def _reconstruct_image(
        self,
        img: np.ndarray,
        fg_indices: np.ndarray,
        bg_indices: np.ndarray,
        fg_nn: np.ndarray,
        im_bg: np.ndarray,
        w: int,
    ) -> np.ndarray:
        """Reconstruct foreground using nearest neighbor background patches."""
        img_recon = np.zeros_like(img, dtype=np.int64)
        counts = np.zeros(img.shape[:2]) + EPS

        fg_x, fg_y = fg_indices[:, 0], fg_indices[:, 1]
        nn_i_j = fg_nn[:, 0]
        cii, cjj = bg_indices[nn_i_j, 0], bg_indices[nn_i_j, 1]

        fg_x = np.clip(fg_x, 0, img.shape[0] - w)
        fg_y = np.clip(fg_y, 0, img.shape[1] - w)
        cii = np.clip(cii, 0, img.shape[0] - w)
        cjj = np.clip(cjj, 0, img.shape[1] - w)

        for i in range(fg_indices.shape[0]):
            img_recon[fg_x[i] : fg_x[i] + w, fg_y[i] : fg_y[i] + w, :] += im_bg[
                cii[i] : cii[i] + w, cjj[i] : cjj[i] + w, :
            ]
            counts[fg_x[i] : fg_x[i] + w, fg_y[i] : fg_y[i] + w] += 1

        counts = np.expand_dims(counts, axis=-1)
        img_recon = np.round(img_recon / counts).astype(np.uint8)

        return img_recon

    def _compute_delta_e2000_matrix(self, img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> np.ndarray:
        """Compute the perceptual color difference (ΔE 2000) between two images.

        Args:
            img1_rgb (np.ndarray): First input image (H x W x 3) in RGB format.
            img2_rgb (np.ndarray): Second input image (H x W x 3) in RGB format.

        Returns:
            np.ndarray: Similarity matrix with values in [0,1] (higher = more similar).
        """
        # Convert RGB to Lab color space
        lab1 = rgb2lab(img1_rgb)
        lab2 = rgb2lab(img2_rgb)

        # Compute ΔE 2000 color difference
        delta_e_matrix = deltaE_ciede2000(lab1, lab2)

        # Normalize ΔE 2000 values to [0,1]
        similarity_matrix = 1 - np.clip(delta_e_matrix / 100, 0, 1)

        return similarity_matrix

    def get_results(self) -> dict:
        """Return the results about camouflage context measure.

        Returns:
            dict(ccm=camouflage_context_measure)
        """
        ccm = np.mean(np.array(self.scores, dtype=TYPE))
        return dict(ccm=ccm)
