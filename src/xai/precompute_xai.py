#!/usr/bin/env python3
"""
Script principal pour pré-calculer les résultats XAI
À exécuter une seule fois pour générer le cache
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
from datetime import datetime
from src.xai.core import XAIEngine
from src.xai.cache_manager import XAICacheManager
from src.models.loader import load_model
from src.data.generator import load_generator

# Configuration
CONFIG = {
    'model_path': 'data/models/best_model.keras',
    'generator_path': 'data/processed/val_gen.pkl',
    'cache_dir': 'data/cache/xai',
    'cases_to_process': [
        {'idx': 0, 'type': 'nv_success', 'note': 'NV correct'},
        {'idx': 2532, 'type': 'vasc_success', 'note': 'VASC correct'},
        {'idx': 387, 'type': 'nv_to_vasc_error', 'note': 'NV confondu avec VASC'},
        {'idx': 2540, 'type': 'vasc_to_nv_error', 'note': 'VASC confondu avec NV'},
        # Ajouter d'autres cas intéressants
    ]
}


def main():
    print("=" * 80)
    print("🔬 PRÉ-CALCUL XAI - GÉNÉRATION DU CACHE")
    print("=" * 80)

    # 1. Initialiser
    cache_mgr = XAICacheManager(CONFIG['cache_dir'])
    cache_mgr.load_cache()

    # 2. Charger le modèle
    print("\n📦 Chargement du modèle...")
    model = load_model(CONFIG['model_path'])

    # 3. Charger le générateur
    print("📊 Chargement du générateur...")
    generator = load_generator(CONFIG['generator_path'])

    # 4. Initialiser le moteur XAI
    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    xai_engine = XAIEngine(model, class_names)

    # 5. Traiter chaque cas
    results = []

    for case_config in CONFIG['cases_to_process']:
        print(f"\n{'=' * 60}")
        print(f"🔄 Traitement: {case_config['type']} (idx={case_config['idx']})")
        print(f"{'=' * 60}")

        try:
            # Charger l'image et les métadonnées
            idx = case_config['idx']
            batch_idx = idx // generator.batch_size
            pos_in_batch = idx % generator.batch_size

            batch = generator[batch_idx]
            x_dict, y_labels = batch

            image = x_dict['image_input'][pos_in_batch]
            meta = x_dict['meta_input'][pos_in_batch]
            true_label = np.argmax(y_labels[pos_in_batch])

            # ANALYSE XAI (sans visualisation)
            start_time = datetime.now()

            xai_result = xai_engine.analyze_image(
                image=image,
                meta=meta,
                true_label=true_label
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            # Ajouter au cache
            case_id = f"{case_config['type']}_{idx}"
            metadata = {
                'processing_time': elapsed,
                'note': case_config['note'],
                'timestamp': datetime.now().isoformat()
            }

            cache_entry = cache_mgr.add_result(
                case_id=case_id,
                image=image,
                xai_result=xai_result,
                metadata=metadata
            )

            # Affichage de confirmation
            pred_label = class_names[xai_result['predicted_class']]
            true_name = class_names[true_label]
            confidence = xai_result['confidence']

            print(f"✅ Ajouté au cache: {case_id}")
            print(f"   Vérité: {true_name} → Prédit: {pred_label} ({confidence:.1%})")
            print(f"   Temps: {elapsed:.1f}s | Consensus: {xai_result['metrics']['consensus_score']:.3f}")

            results.append(cache_entry)

        except Exception as e:
            print(f"❌ Erreur avec {case_config['type']}: {e}")
            import traceback
            traceback.print_exc()

    # 6. Sauvegarder le cache final
    print(f"\n💾 Sauvegarde finale du cache...")
    cache_mgr.save_cache()

    # 7. Générer un rapport
    print(f"\n📊 RAPPORT FINAL:")
    print(f"   Cas traités: {len(results)}/{len(CONFIG['cases_to_process'])}")
    print(f"   Cache size: {os.path.getsize(cache_mgr.cache_file) / 1024:.1f} KB")
    print(f"   Dossier: {CONFIG['cache_dir']}")

    # 8. Créer un fichier de métadonnées pour DVC
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'model_version': '1.0.0',
        'cases_processed': len(results),
        'xai_techniques': ['occlusion', 'integrated_gradients', 'attention_rollout'],
        'cache_structure': {
            'json_cache': cache_mgr.cache_file,
            'numpy_arrays': os.path.join(CONFIG['cache_dir'], 'numpy_arrays'),
            'total_files': sum(len(files) for _, _, files in
                               os.walk(os.path.join(CONFIG['cache_dir'], 'numpy_arrays')))
        }
    }

    metadata_path = os.path.join(CONFIG['cache_dir'], 'cache_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Métadonnées: {metadata_path}")
    print("\n🎉 PRÉ-CALCUL TERMINÉ ! Le cache est prêt pour le backend.")


if __name__ == '__main__':
    main()