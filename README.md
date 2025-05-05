# Personalized Game Recommendations Using Content-Based and Collaborative Filtering

## Overview
This project explores two machine learning approaches, content-based filtering and collaborative filtering, to build a personalized video game recommendation system using player-game interaction data and game metadata. The content-based model uses player genre preferences to recommend unseen games, while the collaborative model leverages a denoising autoencoder to predict likely game interests based on masked game vectors. Both models showed significant improvements after training, with the content-based approach achieving superior performance in Top-5 recommendation accuracy (96.55%) compared to the collaborative model (86.49%). The final system effectively suggests games tailored to user preferences and behaviors, demonstrating the value of combining deep learning with recommender system techniques.

Data set used for this project: https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox/data

## Replication Instructions

Follow the steps below to reproduce the results from this project.

---

### 1. `preprocess.py` 
**Purpose:**  
Cleans raw Steam data by:
- Removing games without genre data  
- Removing players without valid game libraries  
- Filtering out invalid or unusable entries  

**Inputs:**  
- `gaming_datasets/steam/games.csv`  (from dataset)
- `gaming_datasets/steam/purchased_games.csv`  (from dataset)

**Outputs:**  
- `games_genres.csv` — Cleaned game metadata with genre info  
- `player_games.csv` — Cleaned player game libraries  

---

### 2. `collaborative_model.py` 
**Purpose:**  
Trains a **collaborative filtering model** (Denoising Autoencoder):
- Limits to top 1000 most common games  
- Masks a percentage of owned games per user  
- Learns to reconstruct masked inputs  
- Evaluates performance before and after training  

**Inputs:**  
- `player_games.csv`  

**Generated Files:**  
- `player_games_1000.csv`  
- `top_1000_arr.npy`  

**Outputs:**  
- Trained model: `collab_model3_20epoch.h5`  
- Test data:  
  - `collab_data/X_test3.npy`  
  - `collab_data/y_test3.npy`  
  - `collab_data/user_ids_test3.npy`  

---

### 3. `collaborative_eval.py`  
**Purpose:**  
Loads the trained collaborative model and test data to:
- Evaluate classification performance (Accuracy, Precision, Recall, F1 Score)  
- Compute **Top-5 Hit Rate**  
- Print sample user predictions  

**Inputs:**  
- Trained model and test data from `collaborative_model.py`  

**Outputs:**  
- Printed evaluation metrics and Top-5 Hit Rate  
- Example predictions per user  

---

### 4. `content_model.py` 
**Purpose:**  
Trains a **content-based filtering model** using:
- Player genre preferences (built from owned games)  
- Game genre metadata  
- Positive/negative examples based on masked ownership  

**Inputs:**  
- `games_genres.csv`  
- `player_games_1000.csv`  

**Outputs:**  
- Trained model: `content_model3_10epoch.h5`  
- Test data:  
  - `content_model_data/X_test3.npy`  
  - `content_model_data/y_test3.npy`  
  - `content_model_data/xy_map3.npy`  
  - `content_model_data/mlb.pkl` (genre encoder)  

---

### 5. `content_eval.py` 
**Purpose:**  
Loads the trained content-based model and test data to:
- Evaluate classification performance (Accuracy, Precision, Recall, F1 Score)  
- Compute **Top-5 Hit Rate**  
- Print sample user predictions with genre information  

**Inputs:**  
- Trained model and test data from `content_model.py`  

**Outputs:**  
- Printed evaluation metrics and Top-5 Hit Rate  
- Example predictions per user  

---

## Future Directions
To further improve the quality and personalization of recommendations, future work on this project will focus on developing a hybrid model that combines content-based and collaborative filtering approaches into a unified architecture. This hybrid system would aim to capture both the behavioral patterns in user-game interactions (as seen in collaborative filtering) and the rich descriptive features of games such as genre and developer (used in content-based filtering). By leveraging both sources of information, the model has the potential to overcome the limitations of each individual approach and provide more accurate and diverse recommendations. The content-based model can also be enhanced by integrating developer metadata, such as the game studio or publisher. Many players show strong loyalty to specific developers or franchises, and accounting for that can lead to more personalized suggestions. On the collaborative side, incorporating social signals—such as whether a player’s friends own or play a game—could introduce a social dimension to recommendations, capturing the peer influence that often drives game discovery. Additionally, since the dataset includes data from PlayStation, Xbox, and Steam, the model could be extended to support device-aware recommendations. This would allow the system to tailor game suggestions based on the platform a user prefers or owns, helping avoid recommending games that are unavailable on a user’s current system. These ideas include some next steps in building a more flexible, intelligent, and personalized recommendation system that better reflects how people choose and discover games today.

## Contributions

This project was completed independently. All components of the work were carried out by me (Rahul), including:

- **Research and Literature Review**:  
  Identified and analyzed relevant academic papers to guide the design and architecture of both content-based and collaborative filtering models, particularly those utilizing deep learning and recommender systems for games.

- **Dataset Acquisition**:  
  Located and selected a comprehensive multi-platform dataset that includes user behavior data from Steam, PlayStation, and Xbox.

- **Data Cleaning and Preprocessing**:  
  Developed scripts to clean the raw data, including filtering invalid entries, handling missing metadata, parsing game libraries, and preparing the dataset for modeling.

- **Collaborative Filtering Model**:  
  Implemented a denoising autoencoder-based collaborative filtering model inspired by the referenced research. Included logic for masking input data and training the model to predict hidden user-game interactions.

- **Content-Based Filtering Model**:  
  Created a neural network model that uses player genre preferences and game metadata (e.g., genres) to predict game interest. Built logic for masking data, generating training samples, and encoding game features.

- **Model Evaluation**:  
  Conducted pre- and post-training evaluations of both models using metrics such as accuracy, precision, recall, F1 score, and Top-5 hit rate.

- **Visualization and Graphics**:  
  Designed and created clear visual representations of the model architecture, performance metrics, and results for inclusion in the final presentation materials.

- **Poster Creation**:  
  Designed and authored the final project poster, summarizing the research question, methodology, results, and future directions in a visually engaging and informative format.

- **Documentation and Code Structure**:  
  Organized the codebase, added helpful comments, and wrote detailed instructions for reproducing results in the `README.md`.
