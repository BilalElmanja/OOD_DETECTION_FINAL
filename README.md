
# Détection de Données Out-Of-Distribution (OOD)

Ce projet est dédié à l'exploration de l'efficacité de diverses méthodes de détection des données out-of-distribution (OOD) dans les réseaux de neurones. Avec la dépendance croissante envers les modèles d'apprentissage profond dans des applications critiques, assurer que ces modèles peuvent identifier et gérer les entrées OOD—des échantillons de données significativement différents de ceux vus lors de l'entraînement—est primordial pour maintenir la fiabilité et la sécurité.

## Objectifs

L'objectif principal est de réaliser un benchmarking des méthodes suivantes pour la détection OOD :

- Score Maximal des Logits (MLS)
- ODIN : Out-of-DIstribution Detector for Neural Networks
- DkNN : Deep k-Nearest Neighbors
- VIM : Variational Inference for Monte Carlo Objectives
- Détection basée sur l'énergie
- Score d'entropie
- Détection basée sur la distance de Mahalanobis
- Détection basée sur les matrices de Gram

En plus de ces méthodes établies, notre projet introduit une approche novatrice exploitant K-means, PCA et NMF avec des calculs de distance (KNN et Mahalanobis) comme nouvelle méthode pour la détection OOD.


## Installation

Pour installer les dépendances requises pour ce projet, veuillez exécuter la commande suivante :

```bash
pip install -r requirements.txt
```

## Utilisation

Voir les différents notebooks dans le dossier :  `./src/test/`.  

## Structure du Projet

Le projet est organisé comme suit :

- `data/` : Contient les ensembles de données organisés par type (ID, near OOD, far OOD).
- `models/` : Stocke les modèles pré-entraînés sur chaque ensemble de données ID.
- `src/` : Contient le code source pour les méthodes de détection OOD, le code source pour l'utilisation des modèles pre-entrainés et le preprocessing des données avant utilisation.
- `results/` : Dossier pour stocker les résultats des benchmarks.
- `notebooks/` : Jupyter notebooks pour l'analyse exploratoire et les visualisations.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
