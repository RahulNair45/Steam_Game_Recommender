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
To further enhance the accuracy and robustness of game recommendations, future work will focus on developing a **hybrid model** that combines both content-based and collaborative filtering approaches. This unified architecture will aim to leverage the strengths of both methods—user behavior patterns from collaborative filtering and rich metadata (e.g., genres) from content-based filtering—and will be evaluated against the individual models to measure performance gains. Additionally, the content-based model can be improved by integrating **developer-related metadata** such as game studio or publisher, which may capture user preferences for specific creators or franchises. On the collaborative side, incorporating **social signals** like whether a user’s friends own a game, can allow the model to account for **peer influence**, a common driver of game discovery and purchase decisions. These extensions open up exciting opportunities for more personalized and socially-aware recommendations.


