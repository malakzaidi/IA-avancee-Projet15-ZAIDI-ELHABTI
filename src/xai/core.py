"""
XAI Core Engine
Moteur principal des techniques XAI
"""
import numpy as np
import tensorflow as tf
import cv2
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class XAIEngine:
    """Moteur XAI multi-techniques"""

    def __init__(self, model, class_names: List[str]):
        self.model = model
        self.class_names = class_names

    def occlusion_sensitivity(self, image: np.ndarray, meta: np.ndarray,
                              target_class: int, patch_size: int = 32,
                              stride: int = 16) -> np.ndarray:
        """Calcule l'importance par occlusion"""
        logger.info(f"Computing occlusion for class {self.class_names[target_class]}")

        h, w = image.shape[:2]
        base_pred = self._predict_single(image, meta)[target_class]
        importance_map = np.zeros((h, w))

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                img_occluded = image.copy()
                img_occluded[y:y + patch_size, x:x + patch_size] = 0.5

                occ_pred = self._predict_single(img_occluded, meta)[target_class]
                importance = base_pred - occ_pred
                importance_map[y:y + patch_size, x:x + patch_size] = importance

        return self._normalize_map(importance_map)

    def integrated_gradients(self, image: np.ndarray, meta: np.ndarray,
                             target_class: int, steps: int = 50) -> np.ndarray:
        """Calcule les gradients intégrés"""
        baseline = np.zeros_like(image)
        alphas = np.linspace(0, 1, steps)
        integrated_grads = np.zeros_like(image)

        img_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        meta_tensor = tf.convert_to_tensor(np.expand_dims(meta, 0), dtype=tf.float32)

        for alpha in alphas:
            img_interp = baseline + alpha * (image - baseline)
            img_interp_tensor = tf.convert_to_tensor(
                np.expand_dims(img_interp, 0), dtype=tf.float32
            )

            with tf.GradientTape() as tape:
                tape.watch(img_interp_tensor)
                predictions = self.model([img_interp_tensor, meta_tensor], training=False)
                target_output = predictions[0, target_class]

            grads = tape.gradient(target_output, img_interp_tensor)
            integrated_grads += grads[0].numpy()

        integrated_grads = integrated_grads / steps * (image - baseline)
        attribution_map = np.sum(np.abs(integrated_grads), axis=-1)

        return self._normalize_map(attribution_map)

    def attention_rollout(self, image: np.ndarray, meta: np.ndarray,
                          blur_sigma: int = 10) -> np.ndarray:
        """Calcule l'attention rollout simplifié"""
        img_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        meta_tensor = tf.convert_to_tensor(np.expand_dims(meta, 0), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model([img_tensor, meta_tensor], training=False)
            output_sum = tf.reduce_sum(predictions)

        grads = tape.gradient(output_sum, img_tensor)

        if grads is not None:
            grads_np = grads[0].numpy()
            attention = np.sum(np.abs(grads_np), axis=-1)
            attention = gaussian_filter(attention, sigma=blur_sigma)
        else:
            attention = np.zeros(image.shape[:2])

        return self._normalize_map(attention)

    def color_analysis(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyse les couleurs pour NV vs VASC"""
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

        # Masques de couleur
        mask_red1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask_red1, mask_red2) / 255.0

        brown_mask = cv2.inRange(hsv,
                                 np.array([10, 50, 20]),
                                 np.array([20, 255, 200])) / 255.0

        return {'red': red_mask, 'brown': brown_mask}

    def analyze_image(self, image: np.ndarray, meta: np.ndarray,
                      true_label: int) -> Dict:
        """Analyse complète d'une image"""
        # Prédiction
        predictions = self._predict_single(image, meta)
        pred_class = np.argmax(predictions)

        # Générer toutes les cartes XAI
        occlusion_map = self.occlusion_sensitivity(image, meta, pred_class)
        ig_map = self.integrated_gradients(image, meta, pred_class)
        attention_map = self.attention_rollout(image, meta)

        # Cartes spécifiques NV/VASC
        nv_index = self.class_names.index('NV')
        vasc_index = self.class_names.index('VASC')

        occlusion_nv = self.occlusion_sensitivity(image, meta, nv_index)
        occlusion_vasc = self.occlusion_sensitivity(image, meta, vasc_index)

        # Couleurs
        color_masks = self.color_analysis(image)

        # Consensus
        consensus = (occlusion_map + ig_map + attention_map) / 3

        # Retourner les résultats bruts (pas de visualisation)
        return {
            'predictions': predictions.tolist(),
            'predicted_class': int(pred_class),
            'true_class': int(true_label),
            'confidence': float(predictions[pred_class]),
            'correct': bool(pred_class == true_label),

            'heatmaps': {
                'occlusion': occlusion_map,
                'integrated_gradients': ig_map,
                'attention': attention_map,
                'consensus': consensus,
                'occlusion_nv': occlusion_nv,
                'occlusion_vasc': occlusion_vasc,
                'color_red': color_masks['red'],
                'color_brown': color_masks['brown']
            },

            'metrics': {
                'consensus_score': float(consensus.max()),
                'nv_probability': float(predictions[nv_index]),
                'vasc_probability': float(predictions[vasc_index]),
                'red_pixels': int(np.sum(color_masks['red'])),
                'brown_pixels': int(np.sum(color_masks['brown']))
            }
        }

    def _predict_single(self, image: np.ndarray, meta: np.ndarray) -> np.ndarray:
        """Prédiction pour une seule image"""
        img_batch = np.expand_dims(image, axis=0)
        meta_batch = np.expand_dims(meta, axis=0)
        return self.model.predict([img_batch, meta_batch], verbose=0)[0]

    def _normalize_map(self, heatmap: np.ndarray) -> np.ndarray:
        """Normalise une heatmap entre 0 et 1"""
        if heatmap.max() > 0:
            return heatmap / heatmap.max()
        return heatmap