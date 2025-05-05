# Personalized Game Recommendations Using Content-Based and Collaborative Filtering

## Overview
This project explores two machine learning approaches—content-based filtering and collaborative filtering—to build a personalized video game recommendation system using player-game interaction data and game metadata. The content-based model uses player genre preferences to recommend unseen games, while the collaborative model leverages a denoising autoencoder to predict likely game interests based on masked game vectors. Both models showed significant improvements after training, with the content-based approach achieving superior performance in Top-5 recommendation accuracy (96.55%) compared to the collaborative model (86.49%). The final system effectively suggests games tailored to user preferences and behaviors, demonstrating the value of combining deep learning with recommender system techniques.

Data set used for this project: https://www.kaggle.com/datasets/artyomkruglov/gaming-profiles-2025-steam-playstation-xbox/data

## Replication Instructions
1. preprocess.py
Purpose:
Cleans raw Steam data by:
- Removing games without genre data
- Removing players without valid game libraries
- Filtering out invalid or unusable entries

Inputs:
- gaming_datasets/steam/games.csv
- gaming_datasets/steam/purchased_games.csv

Outputs:
- games_genres.csv — metadata for games with genre info
- player_games.csv — cleaned player game libraries

2. collaborative_model.py 
Purpose:
- Trains a collaborative filtering model (Denoising Autoencoder) using masked game vectors:
- Limits to top 1000 most common games
- Masks a percentage of owned games per user
- Learns to reconstruct masked inputs
- Evaluates performance before and after training

Inputs:
- player_games.csv

Generated: player_games_1000.csv, top_1000_arr.npy

Outputs:
- Trained model: collab_model3_20epoch.h5

Test data:
- collab_data/X_test3.npy
- collab_data/y_test3.npy
- collab_data/user_ids_test3.npy

3. collaborative_eval.py 
Purpose:
Loads the trained collaborative model and test data to:
- Evaluate classification performance (accuracy, precision, recall, F1)
- Compute Top-5 Hit Rate
- Print sample user predictions

Inputs:
- Trained model and test data from collaborative_model.py

Output:
- Printed evaluation metrics and Top-5 Hit Rate
- Example predictions per user

4. content_model.py 
Purpose:
Trains a content-based filtering model using:
- Player genre preferences (built from owned games)
- Game genre metadata
- Positive/negative examples based on masked ownership

Inputs:
- games_genres.csv
- player_games_1000.csv

Outputs:
Trained model: content_model3_10epoch.h5

Test data:
- content_model_data/X_test3.npy
- content_model_data/y_test3.npy
- content_model_data/xy_map3.npy
- content_model_data/mlb.pkl (genre encoder)
  
5. content_eval.py 📈
Purpose:
Loads the trained content-based model and test data to:
- Evaluate classification performance (accuracy, precision, recall, F1)
- Compute Top-5 Hit Rate
- Print sample user predictions with genre info

Inputs:
- Trained model and test data from content_model.py

Output:
- Printed evaluation metrics and Top-5 Hit Rate
- Example predictions per user



