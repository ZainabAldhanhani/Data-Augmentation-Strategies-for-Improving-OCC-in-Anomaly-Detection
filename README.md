# Data Augmentation Strategies for Improving One-Class Classification in Anomaly Detection
This repository presents our research on enhancing one-class classification for anomaly detection through advanced data augmentation techniques. We integrate methods such as CutPaste, Elastic Deformations, Fourier Domain Adaptation (FDA), and simulated environmental effects into Deep SVDD, OCGAN, and Autoencoder frameworks to enrich training data diversity. The project systematically evaluates each augmentation—and their combinations—on benchmark datasets, measuring improvements in detection accuracy, false positive rates, and model generalization. By introducing a robust augmentation pipeline across multiple OCC architectures, this work contributes scalable strategies for reducing missed anomalies and false alarms in real-world deployment scenarios.  
  

## Overview

One-Class Classification (OCC) models for anomaly detection often suffer from limited data diversity, which impairs generalization and reduces detection accuracy in real-world applications. Conventional OCC approaches such as Deep SVDD, OCGAN, and Autoencoders are particularly sensitive to unseen variations during deployment, resulting in elevated false positive rates and missed anomalies.

This project explores **advanced data augmentation** techniques to enrich training data and improve OCC model robustness. We implement and systematically evaluate the following augmentation methods on benchmark anomaly detection datasets:

- **CutPaste**  
- **Elastic Deformations**  
- **Fourier Domain Adaptation (FDA)**  
- **Simulated Environmental Effects**  

Our goal is to quantify how each technique (and their combinations) affects anomaly detection performance, false positive rates, and model generalization across Deep SVDD, OCGAN, and Autoencoder frameworks.

## Workflow

Below is the high-level pipeline showing how we train and test each OCC model, with and without data augmentation, before comparing the results:

<p align="center">
  <img src="Figures/Diagram-2.png" alt="Training & Evaluation Workflow" width="700"/>
</p>

1. **Datasets**  
   - UCSD Anomaly Detection
   - Avenue   
   -  MVTec AD  
3. **Training**  
   - Deep SVDD  
   - OCGAN  
   - Autoencoder  
4. **Testing & Metrics**  
   - F1-score  
   - Precision–Recall AUC  
5. **Compare Results**  
   - Measure gains from each augmentation strategy  


## Data Augmentation Techniques

1. **CutPaste**  
   Randomly crops and pastes patches within images to simulate localized defects.  

2. **Elastic Deformations**  
   Applies random grid distortions to mimic non-rigid deformations.  

3. **Fourier Domain Adaptation (FDA)**  
   Swaps low-frequency amplitude components between source and target domains to introduce style variations.  

4. **Simulated Environmental Effects**  
   Adds realistic noise, blur, lighting shifts, and weather artifacts (e.g., rain, fog).  


## Repository Structure

- **`Autoencoder/`**  
  Contains the Autoencoder pipeline for each dataset:  
  - **`UCSD_ped1/`**, **`UCSD_ped2/`**, **`Avenue/`**, **`MVTec/`**  
    - `train_autoencoder.py` — train data with no augmentations
    - `train_autoencoder_CutPaste.py` — train with CutPaste augmentations  
    - `train_autoencoder_ElasticDeformations.py` — train with Elastic Deformations 
    - `train_autoencoder_EnvironmentalEffects.py` — train with Simulated Environmental Effects 
    - `train_autoencoder_FDA.py` — train with Fourier Domain Adaptation (FDA)     


- **`OCGAN/`**  
  Houses the OCGAN workflow per dataset:  
  - **`UCSD_ped1/`**, **`UCSD_ped2/`**, **`Avenue/`**, **`MVTec/`**  
    - `train_autoencoder.py` — train data with no augmentations
    - `train_autoencoder_CutPaste.py` — train with CutPaste augmentations  
    - `train_autoencoder_ElasticDeformations.py` — train with Elastic Deformations 
    - `train_autoencoder_EnvironmentalEffects.py` — train with Simulated Environmental Effects 
    - `train_autoencoder_FDA.py` — train with Fourier Domain Adaptation (FDA)  

- **`SVDD/`**  
  Implements Deep SVDD for each dataset:  
  - **`UCSD_ped1/`**, **`UCSD_ped2/`**, **`Avenue/`**, **`MVTec/`**  
    - `train_autoencoder.py` — train data with no augmentations
    - `train_autoencoder_CutPaste.py` — train with CutPaste augmentations  
    - `train_autoencoder_ElasticDeformations.py` — train with Elastic Deformations 
    - `train_autoencoder_EnvironmentalEffects.py` — train with Simulated Environmental Effects 
    - `train_autoencoder_FDA.py` — train with Fourier Domain Adaptation (FDA)  

- **`Demo/`**  
  A demo file (notebook) with sample inputs and outputs.

- **`README.md`**  
  Project overview, installation, usage, and structure.

- **`requirements.txt`**  
  Python dependencies and version requirements.


### Dataset

This project leverages several widely-used benchmark datasets in the anomaly detection domain:

- **[UCSD Anomaly Detection Dataset](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm)**  
  Grayscale surveillance videos of pedestrian walkways recorded by a fixed elevated camera, split into Ped1 (34 train / 36 test) and Ped2 (16 train / 12 test) clips labeled per frame.

- **[CUHK Avenue Dataset](https://www.kaggle.com/datasets/vibhavvasudevan/avenue)**  
  16 training videos of normal campus-avenue activity and 21 testing videos (15 328 train / 15 324 test frames) containing anomalies such as wrong-direction walking, loitering, and novel objects.

- **[MVTec Anomaly Detection (MVTec AD)](https://www.kaggle.com/datasets/ipythonx/mvtec-ad?select=toothbrush)**  
  5354 high-resolution images across 15 industrial object and texture categories, with defect-free images for training and both normal and defective samples (with pixel-precise masks) for testing.


## Getting Started

### Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/occ-data-augmentation.git
   cd occ-data-augmentation
    ```

2. **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```
3. **Add the Datasets**
   
   Create a data/ folder in the project root

   Download or copy the datasets into `data/` folder
   ```bash
    mkdir -p data/
    ```
## Train and Evaluate the Models

Each model folder contains one “vanilla” training script plus one for each augmentation strategy:

1. **Autoencoder**
    ```bash
    # move into the dataset folder
    cd Autoencoder/<Dataset>
    # e.g. cd Autoencoder/Avenue
    
    # train without augmentation
    python train_autoencoder.py --dataset_path /path/to/Avenue
    
    # -- or with a specific augmentation --
    python train_autoencoder_CutPaste.py --dataset_path /path/to/Avenue
    python train_autoencoder_ElasticDeform.py --dataset_path /path/to/Avenue
    python train_autoencoder_FDA.py --dataset_path /path/to/Avenue
    python train_autoencoder_Environmental.py --dataset_path /path/to/Avenue
    ```
    For MVTec, you can point --dataset_path at any category folder(e.g. toothbrush, wood, grid):
    ```bash
    cd Autoencoder/Mvtec
    python train_autoencoder_FDA.py --dataset_path /path/to/mvtec_ad/toothbrush
    
    ```
2. **SVDD**
    ```bash
    # move into the dataset folder
    cd SVDD/<Dataset>
    # e.g. cd SVDD/Avenue
    
    # train without augmentation
    python train_SVDD.py --dataset_path /path/to/Avenue
    
    # -- or with a specific augmentation --
    python train_SVDD_CutPaste.py --dataset_path /path/to/Avenue
    python train_SVDD_ElasticDeform.py --dataset_path /path/to/Avenue
    python train_SVDD_FDA.py --dataset_path /path/to/Avenue
    python train_SVDD_Environmental.py --dataset_path /path/to/Avenue
    ```
    For MVTec, you can point --dataset_path at any category folder(e.g. toothbrush, wood, grid):
    ```bash
    cd SVDD/Mvtec
    python train_SVDD_FDA.py --dataset_path /path/to/mvtec_ad/toothbrush
    
    ```
3. **OCGAN**
    ```bash
    # move into the dataset folder
    cd OCGAN/<Dataset>
    # e.g. cd OCGAN/Avenue
    
    # train without augmentation
    python train_OCGAN.py --dataset_path /path/to/Avenue
    
    # -- or with a specific augmentation --
    python train_OCGAN_CutPaste.py --dataset_path /path/to/Avenue
    python train_OCGAN_ElasticDeform.py --dataset_path /path/to/Avenue
    python train_OCGAN_FDA.py --dataset_path /path/to/Avenue
    python train_OCGAN_Environmental.py --dataset_path /path/to/Avenue
    ```
    For MVTec, you can point --dataset_path at any category folder(e.g. toothbrush, wood, grid):
    ```bash
    cd OCGAN/Mvtec
    python train_OCGAN_FDA.py --dataset_path /path/to/mvtec_ad/toothbrush
    
    ```
## Demo 
We provide a ready-to-run notebook for demonstrating the workflow:

`demo/SVDD_Avenue.ipynb`

**This notebook includes:**
- Loading and preprocessing of the Avenue dataset
- Training the SVDD on normal and augmented dats
- Visualizing sample inputs and reconstructed outputs
- Computing F1-score and PR-AUC for anomaly detection

**Run the demo locally:**

    ```bash
        # Step 1: Clone the repository
        git clone https://github.com/ZainabAldhanhani/Enhancing-Deepfake-Detection.git
        cd Enhancing-Deepfake-Detection
        
        # Step 2: (Optional) Create and activate a virtual environment
        python -m venv venv
        source venv/bin/activate  
        
        # Step 3: Install requirements
        pip install -r requirements.txt
        
        # Step 4: Launch the demo notebook
        jupyter notebook demo/SVDD_Avenue.ipynb
        ```



  ## Results Summary

We evaluated three One-Class Classification (OCC) models — Autoencoder, OCGAN, and Deep SVDD — across three benchmark datasets: UCSD, Avenue, and MVTec.
The models were tested with and without five augmentation strategies: CutPaste, Elastic Deformations, Fourier Domain Adaptation (FDA), and Simulated Environmental Effects.

### Key Findings:

**Autoencoder:**

Elastic Deformations consistently achieved the highest gains in F1-score and PR-AUC (e.g., +13.17% F1 on UCSD Ped1 and +15.73% PR-AUC on Ped2).
FDA showed strong performance in MVTec (e.g., best F1 on Grid and Wood).

**OCGAN:**

Results were model- and dataset-dependent. While CutPaste and Simulated Effects were beneficial in some cases, Elastic Deformations and FDA led to the most consistent improvements.
Toothbrush PR-AUC improved by +26.29% with simulated effects.

**Deep SVDD:**

FDA and Simulated Environmental Effects provided substantial improvements across MVTec datasets.
In the Wood category, Simulated Environmental Effects boosted PR-AUC by +14.61%, and FDA improved F1-score by +10.64%.
