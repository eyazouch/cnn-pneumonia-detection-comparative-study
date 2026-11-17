# Détection de pneumonie sur radiographies thoraciques

Étude comparative de réseaux de neurones convolutifs pour classer des radiographies thoraciques pédiatriques en deux classes, normale ou pneumonie. Le projet compare un CNN conçu de zéro à des modèles pré-entraînés sur ImageNet (VGG16, DenseNet121), utilisés en extraction de caractéristiques puis en fine-tuning. Une application Streamlit permet de tester les modèles sur de nouvelles images.

Projet académique réalisé en cycle d'ingénieur à la Faculté des Sciences de Sfax.

## Résultats

Évaluation sur 608 images de test :

| Modèle | Accuracy | Precision | Recall | F1-score | AUC-ROC |
|---|---|---|---|---|---|
| CNN personnalisé | 0,868 | 0,889 | 0,898 | 0,894 | 0,925 |
| DenseNet121 | 0,893 | 0,899 | 0,930 | 0,915 | 0,959 |
| DenseNet121 fine-tuné | 0,903 | **0,907** | 0,939 | 0,922 | **0,965** |
| VGG16 | **0,905** | 0,905 | 0,944 | 0,924 | 0,956 |
| VGG16 fine-tuné | **0,905** | 0,897 | **0,955** | **0,925** | 0,959 |

Le VGG16 fine-tuné est retenu : il obtient le meilleur rappel et le meilleur F1-score. En contexte médical, le rappel est prioritaire, puisqu'un cas de pneumonie non détecté coûte plus cher qu'une fausse alerte. Les modèles pré-entraînés dépassent nettement le CNN conçu de zéro, et le fine-tuning apporte un gain modeste mais régulier.

Des cartes Grad-CAM, calculées sur le CNN personnalisé, montrent les zones de l'image sur lesquelles le modèle fonde sa décision.

## Méthodologie

- **Données** : redimensionnement en 224 x 224, normalisation, augmentation (rotation, zoom, retournement horizontal).
- **CNN personnalisé** : blocs convolutifs avec pooling et dropout, utilisé comme référence.
- **Transfert d'apprentissage** : VGG16 et DenseNet121 gelés, avec une tête de classification dense entraînée sur le jeu de données.
- **Fine-tuning** : dégel des derniers blocs convolutifs et réentraînement avec un taux d'apprentissage réduit.
- **Évaluation** : matrices de confusion, courbes ROC, comparaison des métriques, Grad-CAM.

## Jeu de données

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), Kaggle. Environ 5 000 images d'entraînement, réparties en deux classes (`NORMAL`, `PNEUMONIA`). Le jeu de données et les modèles entraînés ne sont pas inclus dans le dépôt.

## Exécution

```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn pillow jupyter streamlit
```

1. Télécharger le jeu de données et le placer dans `chest_xray/` (sous-dossiers `train`, `val`, `test`).
2. Exécuter `Project Code.ipynb` pour entraîner les modèles. Ils sont enregistrés dans `models/` avec le tableau comparatif `models_comparison.csv`.
3. Lancer l'application de test :

```bash
streamlit run app.py
```

L'application, disponible sur http://localhost:8501, permet de charger une radiographie, de choisir un modèle et d'afficher la classe prédite avec son niveau de confiance.

## Limites

- Jeu de données de taille réduite, issu d'un seul établissement et limité aux enfants.
- Modèle non validé cliniquement : ce travail a une vocation pédagogique et ne doit pas servir à poser un diagnostic.

## Références

- K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition*, 2014.
- G. Huang et al. *Densely Connected Convolutional Networks*, 2017.
- R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, 2017.
