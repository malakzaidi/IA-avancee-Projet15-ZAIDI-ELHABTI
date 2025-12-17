"""
Gestion du cache XAI pré-calculé
"""
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, List, Any
import os
import pickle
import hashlib


class XAICacheManager:
    """Gestionnaire du cache XAI"""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "xai_cache.json")
        self.cache = {}

    def load_cache(self):
        """Charge le cache depuis le fichier"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
            print(f"✅ Cache chargé: {len(self.cache)} entrées")
        else:
            print("⚠ Cache vide, création d'un nouveau")

    def save_cache(self):
        """Sauvegarde le cache"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
        print(f"💾 Cache sauvegardé: {self.cache_file}")

    def add_result(self, case_id: str, image: np.ndarray,
                   xai_result: Dict, metadata: Dict = None):
        """Ajoute un résultat au cache"""

        # Générer un hash unique pour l'image
        image_hash = self._hash_image(image)

        # Préparer les données
        cache_entry = {
            'case_id': case_id,
            'image_hash': image_hash,
            'prediction': {
                'true_label': xai_result.get('true_class'),
                'predicted_label': xai_result.get('predicted_class'),
                'confidence': xai_result.get('confidence'),
                'correct': xai_result.get('correct'),
                'all_probabilities': xai_result.get('predictions'),
                'nv_probability': xai_result.get('metrics', {}).get('nv_probability'),
                'vasc_probability': xai_result.get('metrics', {}).get('vasc_probability')
            },
            'xai_metrics': xai_result.get('metrics', {}),
            'metadata': metadata or {},
            'encoded_data': self._encode_for_cache(image, xai_result['heatmaps'])
        }

        self.cache[case_id] = cache_entry

        # Sauvegarder aussi les arrays numpy séparément (pour plus tard)
        self._save_numpy_arrays(case_id, xai_result['heatmaps'])

        return cache_entry

    def get_case(self, case_id: str) -> Dict:
        """Récupère un cas du cache"""
        if case_id in self.cache:
            return self.cache[case_id]
        return None

    def get_all_cases(self) -> List[str]:
        """Retourne la liste de tous les IDs de cas"""
        return list(self.cache.keys())

    def _encode_for_cache(self, image: np.ndarray, heatmaps: Dict) -> Dict:
        """Encode les données pour le cache JSON"""

        # Convertir l'image en base64
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Pour les heatmaps, on peut sauvegarder des miniatures
        encoded_heatmaps = {}
        for name, heatmap in heatmaps.items():
            # Sauvegarder en PNG base64 pour visualisation rapide
            hm_normalized = (heatmap * 255).astype(np.uint8)
            hm_pil = Image.fromarray(hm_normalized)
            buffered = BytesIO()
            hm_pil.save(buffered, format="PNG")
            encoded_heatmaps[f"{name}_thumbnail"] = base64.b64encode(
                buffered.getvalue()
            ).decode()

            # Sauvegarder les statistiques (pas l'array complet)
            encoded_heatmaps[f"{name}_stats"] = {
                'max': float(heatmap.max()),
                'mean': float(heatmap.mean()),
                'std': float(heatmap.std())
            }

        return {
            'original_image': img_str,
            'heatmaps': encoded_heatmaps
        }

    def _save_numpy_arrays(self, case_id: str, heatmaps: Dict):
        """Sauvegarde les arrays numpy pour une utilisation ultérieure"""
        case_dir = os.path.join(self.cache_dir, "numpy_arrays", case_id)
        os.makedirs(case_dir, exist_ok=True)

        for name, array in heatmaps.items():
            np.save(os.path.join(case_dir, f"{name}.npy"), array)

    def _hash_image(self, image: np.ndarray) -> str:
        """Génère un hash pour l'image"""
        # Convertir en bytes
        img_bytes = image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()[:16]