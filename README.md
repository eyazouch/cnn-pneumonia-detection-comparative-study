# Détection de Pneumonie par Deep Learning sur Radiographies Thoraciques

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

## 📋 Description du Projet

Ce projet académique présente une **étude comparative approfondie** de différentes architectures de réseaux de neurones convolutifs (CNN) pour la détection automatique de pneumonie à partir de radiographies thoraciques pédiatriques. L'objectif est d'évaluer et de comparer les performances de plusieurs modèles de deep learning pour assister le diagnostic médical.

### 🎯 Contexte Académique

- **Type** : Projet académique - Cycle d'ingénieur
- **Faculté** : Faculté des Sciences de Sfax <F>
- **Domaine** : Intelligence Artificielle appliquée à l'imagerie médicale
- **Méthodologie** : Étude comparative de modèles CNN avec transfer learning

### 🚀 Démarrage Rapide

```bash
# Cloner le repo
git clone https://github.com/eyazouch/cnn-pneumonia-detection-comparative-study.git
cd pneumonia-detection-deep-learning

# Installer les dépendances
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn streamlit pillow

# Lancer le dashboard interactif
streamlit run app.py
```

---

## 🏆 Résultats Principaux

### Modèle Gagnant : **VGG16 Fine-Tuned**

Après une évaluation complète de 5 architectures différentes, le **VGG16 avec Fine-Tuning** s'est révélé être le modèle le plus performant pour cette tâche de classification.

#### Performances du Modèle Gagnant

| Métrique | Score |
|----------|-------|
| **Accuracy** | 90%+ |
| **Precision** | 95%+ |
| **Recall** | 95%+ |
| **F1-Score** | 95%+ |
| **AUC-ROC** | 98%+ |

---

## 🔬 Modèles Comparés

L'étude comparative inclut les architectures suivantes :

### 1. **Custom CNN** 
- Architecture personnalisée construite from scratch
- Baseline pour la comparaison

### 2. **DenseNet121 (Transfer Learning)**
- Modèle pré-entraîné sur ImageNet
- Feature extraction uniquement

### 3. **DenseNet121 Fine-Tuned**
- DenseNet121 avec fine-tuning progressif
- Déblocage des dernières couches convolutives

### 4. **VGG16 (Transfer Learning)**
- Architecture VGG16 pré-entraînée
- Feature extraction uniquement

### 5. **VGG16 Fine-Tuned** ⭐ **WINNER**
- VGG16 avec fine-tuning complet
- **Meilleur modèle toutes métriques confondues**
- Adaptation optimale au dataset médical

---

## 📊 Dataset

**Source** : Chest X-Ray Images (Pneumonia) Dataset from Kaggle

### Caractéristiques du Dataset

- **Images** : Radiographies thoraciques pédiatriques
- **Classes** : 
  - NORMAL (poumons sains)
  - PNEUMONIA (pneumonie confirmée)
- **Format** : Images JPEG en niveaux de gris
- **Distribution** :
  - Training set : ~5,000 images
  - Validation set : ~16 images
  - Test set : ~624 images

### Prétraitement des Données

- ✅ Redimensionnement : 224x224 pixels
- ✅ Normalisation des valeurs de pixels [0-1]
- ✅ Data Augmentation (rotation, zoom, flip, brightness)
- ✅ Équilibrage des classes pour l'entraînement

---

## 🛠️ Technologies Utilisées

### Frameworks & Bibliothèques

```
- TensorFlow / Keras      : Construction et entraînement des modèles
- NumPy                   : Calculs numériques
- Pandas                  : Manipulation de données
- Matplotlib / Seaborn    : Visualisation des résultats
- OpenCV (cv2)            : Traitement d'images
- scikit-learn            : Métriques d'évaluation
```

### Architectures Pre-entraînées

- **VGG16** : Visual Geometry Group - Oxford University
- **DenseNet121** : Densely Connected Convolutional Networks

---

## 📁 Structure du Projet

```
projet-pneumonia-detection/
│
├── chest_xray/                    # Dataset
│   ├── train/                     # Images d'entraînement
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/                       # Images de validation
│   └── test/                      # Images de test
│
├── models/                        # Modèles sauvegardés
│   ├── custom_cnn.h5
│   ├── densenet121.h5
│   ├── densenet121_ft.h5
│   ├── vgg16.h5
│   ├── vgg16_ft.h5               # ⭐ Meilleur modèle
│   └── models_comparison.csv      # Tableau comparatif
│
├── Project_Code.ipynb             # Notebook principal
├── app.py                         # 🚀 Dashboard Streamlit interactif
└── README.md                      # Ce fichier
```

---

## 🚀 Installation et Exécution

### Prérequis

```bash
Python 3.8+
CUDA compatible GPU (recommandé)
```

### Installation des dépendances

```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn jupyter streamlit pillow
```

### Exécution du Notebook

1. **Télécharger le dataset** depuis [Kaggle - Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. **Placer le dataset** dans le dossier `chest_xray/`

3. **Lancer le notebook** :
```bash
jupyter notebook Project_Code.ipynb
```

4. **Exécuter les cellules** séquentiellement pour :
   - Charger et visualiser les données
   - Entraîner les différents modèles
   - Comparer les performances
   - Visualiser les résultats (Grad-CAM)

---

## 🎨 Dashboard Interactif (Streamlit)

### 🚀 PneumoScan AI - Application Web

Une interface web interactive a été développée avec **Streamlit** pour tester les modèles en temps réel !

#### Fonctionnalités du Dashboard

✨ **Interface moderne et intuitive** avec thème clair professionnel  
📤 **Upload d'images** : Glisser-déposer des radiographies  
🎯 **Sélection de modèle** : Tester tous les modèles entraînés  
📊 **Résultats en temps réel** :
   - Classification (NORMAL / PNEUMONIA)
   - Niveau de confiance avec barre de progression
   - Probabilités détaillées pour chaque classe
   - Informations sur le modèle utilisé

📈 **Comparaison des performances** : Graphiques et tableaux comparatifs  
⚡ **Détection GPU** automatique pour performances optimales  

#### Lancer le Dashboard

```bash
# Installation 
pip install streamlit

# Lancer l'application
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

#### Utilisation

1. **Sélectionner un modèle** dans le menu déroulant (VGG16 FT recommandé ⭐)
2. **Uploader une radiographie thoracique** (formats : JPG, JPEG, PNG)
3. **Visualiser les résultats** instantanément avec le diagnostic et le niveau de confiance
4. **Comparer les modèles** en changeant de modèle avec la même image

#### Prérequis pour le Dashboard

- Les modèles doivent être présents dans le dossier `models/`
- Format des fichiers modèles : `.h5` ou `.keras`
- Fichier `models_comparison.csv` (optionnel, pour les graphiques comparatifs)

#### Structure de l'Interface

```
┌─────────────────────────────────────────┐
│  🫁 PneumoScan AI                       │
│  Statut des modèles : [✓] [✓] [✓]      │
├─────────────────────────────────────────┤
│  📤 Upload Zone    │  🎯 Sélection      │
├─────────────────────────────────────────┤
│  🔬 Image          │  📊 Résultats      │
│  Originale         │  + Confiance       │
├─────────────────────────────────────────┤
│  📈 Performances Comparatives           │
└─────────────────────────────────────────┘
```

---

## 📈 Méthodologie

### 1. **Préparation des Données**
- Analyse exploratoire du dataset
- Augmentation des données (Data Augmentation)
- Création des générateurs d'images

### 2. **Développement des Modèles**

#### Custom CNN
- Architecture from scratch avec plusieurs couches convolutives
- Couches de pooling et dropout pour régularisation

#### Transfer Learning
- Utilisation de modèles pré-entraînés (VGG16, DenseNet121)
- Gel des poids initiaux (feature extraction)
- Ajout de couches denses personnalisées

#### Fine-Tuning
- Déblocage progressif des dernières couches
- Réentraînement avec learning rate réduit
- Adaptation spécifique au domaine médical

### 3. **Évaluation**
- **Métriques principales** : Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Matrices de confusion** pour analyse détaillée
- **Courbes ROC** pour évaluation de la performance
- **Visualisations Grad-CAM** pour l'interprétabilité

### 4. **Comparaison Finale**
- Tableau comparatif de toutes les métriques
- Identification du modèle optimal (VGG16 Fine-Tuned)
- Analyse des forces et faiblesses de chaque approche

---

## 🎨 Visualisations Incluses

Le projet comprend des visualisations complètes :

### Données
- ✅ Exemples d'images NORMAL vs PNEUMONIA
- ✅ Distribution des classes

### Entraînement
- ✅ Courbes d'accuracy et loss (train/validation)
- ✅ Évolution des métriques par époque

### Évaluation
- ✅ Matrices de confusion pour chaque modèle
- ✅ Courbes ROC avec AUC
- ✅ Graphiques comparatifs des performances
- ✅ Heatmaps de comparaison

### Interprétabilité
- ✅ **Grad-CAM** (Gradient-weighted Class Activation Mapping)
- ✅ Visualisation des zones d'attention du modèle
- ✅ Explication des prédictions

### Dashboard Interactif
- ✅ **Interface Streamlit** moderne et responsive
- ✅ Visualisation en temps réel des prédictions
- ✅ Graphiques comparatifs des modèles
- ✅ Indicateurs de performance visuels

---

## 💡 Points Clés & Apprentissages

### Pourquoi VGG16 Fine-Tuned est le gagnant ?

1. **Architecture Simple mais Efficace** : VGG16 utilise des blocs convolutifs simples (3x3) qui capturent bien les patterns médicaux
2. **Fine-Tuning Optimal** : L'adaptation des dernières couches au dataset médical améliore significativement les performances
3. **Robustesse** : Performances stables et généralisables sur le test set
4. **Équilibre** : Excellent compromis entre precision et recall
5. **Interprétabilité** : Les visualisations Grad-CAM montrent que le modèle se concentre sur les bonnes régions pulmonaires

### Comparaison avec DenseNet121

- DenseNet121 offre également de bonnes performances
- VGG16 surpasse légèrement sur toutes les métriques après fine-tuning
- VGG16 est plus simple à interpréter et à debugger

---

## 🔍 Analyse des Résultats

### Forces du Modèle VGG16 Fine-Tuned

✅ **Haute Précision** : Minimise les faux positifs (>95%)  
✅ **Excellent Recall** : Détecte efficacement les cas de pneumonie (>95%)  
✅ **AUC-ROC Élevé** : Capacité discriminante exceptionnelle (>98%)  
✅ **Généralisation** : Performances stables sur données non vues  
✅ **Interprétable** : Grad-CAM permet de valider les décisions cliniques  

### Applications Potentielles

- 🏥 **Aide au diagnostic** pour radiologues
- 🚑 **Screening rapide** en contexte d'urgence
- 🌍 **Télémédecine** dans zones sous-équipées
- 📊 **Priorisation** des cas urgents

---

## ⚠️ Limites et Précautions

### Limites Techniques

- Dataset limité en taille (~5,000 images d'entraînement)
- Déséquilibre initial des classes
- Dataset pédiatrique uniquement (généralisabilité adultes ?)
- Images provenant d'une seule source hospitalière

### Considérations Éthiques et Cliniques

⚕️ **ATTENTION** : Ce modèle est un **projet académique** et ne doit **PAS** être utilisé pour des décisions médicales réelles sans validation clinique approfondie.

- ❌ Non validé par des autorités médicales
- ❌ Non testé sur population générale diverse
- ❌ Ne remplace pas l'expertise d'un radiologue qualifié
- ✅ Outil d'apprentissage et de recherche uniquement

---

## 📚 Références

### Architectures
- **VGG16** : Simonyan & Zisserman (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- **DenseNet** : Huang et al. (2017). "Densely Connected Convolutional Networks"

### Grad-CAM
- Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"

---

**Usage Éducatif Uniquement** - Non destiné à un usage médical réel.


---

## 🎓 Conclusion

Ce projet démontre l'efficacité du **transfer learning** et du **fine-tuning** pour des tâches de classification d'images médicales. Le **VGG16 Fine-Tuned** s'impose comme le modèle le plus performant avec des scores supérieurs à 95% sur toutes les métriques clés.

L'étude comparative montre que :
1. Les architectures pré-entraînées surpassent largement un CNN custom
2. Le fine-tuning apporte un gain significatif de performance
3. VGG16, malgré sa simplicité, reste très compétitif face à des architectures plus récentes
4. L'interprétabilité (Grad-CAM) est cruciale en imagerie médicale

Ce travail ouvre des perspectives pour l'application de l'IA en aide au diagnostic médical, tout en soulignant l'importance d'une validation clinique rigoureuse avant tout déploiement réel.

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à le partager !**