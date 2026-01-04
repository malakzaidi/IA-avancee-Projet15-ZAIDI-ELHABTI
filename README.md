# IA Explicable Multi-Modal pour la Classification Dermatologique

## Aperçu du Projet
Ce projet propose une solution d'intelligence artificielle explicable et multi-modale pour la classification automatisée de huit types de lésions cutanées à partir du dataset ISIC 2019. Face aux défis posés par le fort déséquilibre des classes et la complexité visuelle des lésions, une architecture hybride a été développée, combinant un réseau de neurones convolutif (CNN) basé sur DenseNet121 avec des données cliniques (âge, sexe, localisation).

<img width="703" height="752" alt="image" src="https://github.com/user-attachments/assets/d8bcb627-f860-4ab4-a838-f81738fce741" />


L'approche repose sur un pipeline de prétraitement rigoureux (élimination des poils, normalisation colorimétrique) et une stratégie d'entraînement avancée intégrant le Transfer Learning, des fonctions de perte adaptées (Focal Loss) et des techniques d'augmentation de données (MixUp, CutMix) pour améliorer la généralisation. Une attention particulière a été portée à la transparence du modèle via des techniques d'IA Explicable (XAI), notamment Grad-CAM et l'Occlusion Sensitivity.

Les résultats démontrent une amélioration significative du rappel (Recall) sur les classes critiques et fournissent des visualisations validant la pertinence clinique des zones analysées par le modèle.

**Mots-clés :** Classification d'images médicales, ISIC 2019, DenseNet121, Transfer Learning, Fusion Multimodale, XAI, Déséquilibre de classes.

## Fonctionnalités Principales
- **Classification Multi-Modale :** Fusion d'images dermoscopiques et de métadonnées cliniques (âge, sexe, localisation) pour une précision accrue.
- **Optimisation pour Déséquilibre :** Utilisation de Focal Loss et d'augmentations avancées (MixUp, CutMix) pour booster le rappel sur les classes rares comme le mélanome.
- **Explicabilité (XAI) :** Intégration de Grad-CAM, Occlusion Sensitivity, Integrated Gradients et LIME pour des visualisations interactives (cartes de chaleur 3D via Plotly) expliquant les décisions du modèle.
- **Pipeline de Prétraitement :** Élimination des poils, normalisation colorimétrique, standardisation géométrique pour gérer l'hétérogénéité des données ISIC.
- **Interface Utilisateur :** Application web Flask avec frontend pour soumission d'images, visualisation des résultats et génération de rapports PDF incluant avertissements légaux.
- **Score de Confiance Composite :** Agrégation de la probabilité du modèle et de la cohérence des explications XAI pour des alertes cliniques (ex. : "Confiance Limite - Biopsie nécessaire").
- **MLOps Intégré :** Utilisation de DVC pour le versioning des données, MLflow pour le tracking des expériences, et automatisation des rapports.

## Technologies Utilisées
- **IA/ML :** TensorFlow/Keras (DenseNet121 pré-entraîné sur ImageNet), OpenCV (prétraitement), Scikit-learn (métriques, clustering).
- **XAI :** Grad-CAM, Occlusion Sensitivity, LIME, Integrated Gradients.
- **Backend :** Flask (API et contrôleur MVC).
- **Frontend :** HTML/CSS/JS avec Plotly pour visualisations interactives 3D.
- **Rapports :** ReportLab pour génération PDF.
- **MLOps :** MLflow (tracking), DVC (versioning des données et modèles).
- **Autres :** Pandas/NumPy (analyse exploratoire), Matplotlib/Seaborn (visualisations EDA).

## Architecture du Modèle
L'architecture est hybride et multi-input, articulée autour de deux branches :

- **Branche Visuelle :** DenseNet121 pour extraction de caractéristiques des images (redimensionnées à 224x224), suivi d'un Global Average Pooling.
- **Branche Métadonnées :** MLP (couches denses avec Dropout) pour traiter les variables cliniques encodées (11 features après one-hot encoding).
- **Fusion :** Concaténation des sorties des branches, suivie de couches denses pour classification finale (8 classes).

Le backbone DenseNet121 est initialement gelé pour le Transfer Learning, puis fine-tuné sélectivement.

![Figure 17: Architecture du modèle](path/to/architecture_model.png)

![Figure 16: Architecture globale du projet](path/to/architecture_global.png)

## Analyse Exploratoire des Données (EDA)
- **Dataset ISIC 2019 :** 25 331 images de 8 classes de lésions cutanées, avec métadonnées cliniques.
- **Déséquilibre :** Classes dominantes (ex. : Nævus ~50%) vs. rares (ex. : Mélanome ~18%).
- **Visualisations :** Analyse colorimétrique (signatures spectrales), t-SNE pour structure visuelle, PCA/K-Means pour profilage démographique.

## Processus de Prétraitement
- **Images :** Redimensionnement, élimination des poils (algorithme DullRazor), normalisation colorimétrique.
- **Métadonnées :** Encodage one-hot pour sexe et localisation, normalisation pour âge, gestion des valeurs manquantes.

## Stratégie d'Entraînement
- **Phases :** Gel initial du backbone, puis fine-tuning progressif.
- **Perte :** Focal Loss pour focaliser sur les classes difficiles.
- **Augmentation :** Rotations, flips, MixUp/CutMix pour robustesse.
- **Optimisation :** Adam avec scheduler (ReduceLROnPlateau), Early Stopping.

## Explicabilité (XAI)
- **Méthodes :** Grad-CAM pour heatmaps, Occlusion Sensitivity pour masquage, LIME pour attributions locales.
- **Visualisations :** Cartes de chaleur interactives 3D, analyse qualitative des succès/échecs (ex. : focus sur bordures irrégulières pour mélanome).

![Figure 29: Analyse XAI de deux nævus correctement classifiés](path/to/xai_nevus.png)

## Résultats
Les résultats montrent une précision globale de 89,2 %, avec un rappel amélioré sur les classes critiques (ex. : Mélanome).

| Métrique | Valeur Globale | Exemple (Mélanome) |
|----------|----------------|---------------------|
| Précision | 89.2%         | -                   |
| Rappel (Recall) | -           | Amélioration significative |
| F1-Score | -             | -                   |

Analyse des erreurs via XAI révèle des confusions visuelles (ex. : nævus vs. vasculaire dues à couleurs similaires).

![Figure 23: Visualisation de Recall par classe](path/to/recall_classes.png)

## Gestion du Projet
Méthodologie Scrumban : Sprints pour modélisation, Kanban pour tâches EDA et XAI. Outils : Git, Trello, Google Colab.

![Figure 13: Diagramme de Gantt du projet](path/to/gantt.png)

## Installation et Exécution
1. Cloner le repository : `git clone https://github.com/votre-repo/projet15-ia-explicable-dermatologie.git`
2. Installer les dépendances : `pip install -r requirements.txt` (TensorFlow, Keras, Flask, Plotly, etc.)
3. Lancer MLflow : `mlflow ui --port 5000`
4. Démarrer l'application Flask : `python app.py`
5. Accéder à l'interface : `http://localhost:5000`

**Prérequis :** Python 3.8+, GPU recommandé pour entraînement, dataset ISIC 2019 (disponible sur Kaggle).

## Auteurs
- ZAIDI Malak
- ELHABTI Fatiha

**Encadrant :** Pr. HAMIDA Soufiane

**Date :** Décembre 2025

Ce projet est réalisé dans le cadre du Master SDIA – Module IA Avancée (Année 2025-2026). Contributions bienvenues via pull requests !
