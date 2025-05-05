import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict

# Load saved model
model = load_model('content_model3_10epoch.h5')

# Load saved test data
X_test = np.load('content_model_data/X_test3.npy')
y_test = np.load('content_model_data/y_test3.npy')
xy_map = np.load('content_model_data/xy_map3.npy', allow_pickle=True)

print("\post train eval:\n")
# Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Get predicted probabilities from the model
y_pred_probs = model.predict(X_test).flatten()

# Convert probabilities to binary predictions using a threshold
y_pred = (y_pred_probs >= 0.5).astype(int)

# Calculate additional metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\ndone with post eval\n")

# Predict probabilities
y_probs = model.predict(X_test).flatten()

# Group predictions by player
player_to_predictions = defaultdict(list)
for i, (player_id, game_id) in enumerate(xy_map):
    player_to_predictions[player_id].append((game_id, y_probs[i], y_test[i]))

# Evaluate Top-N Hit Rate
N = 5
hits = 0
total_users = 0

for player_id, predictions in player_to_predictions.items():
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    top_n = sorted_preds[:N]
    hit = any(label == 1 for _, _, label in top_n)
    hits += int(hit)
    total_users += 1

hit_rate = hits / total_users
print(f"Top-{N} Hit Rate: {hit_rate:.4f}")

# --- NEW: Calculate average percent of owned games per user ---
user_counts = defaultdict(lambda: [0, 0])  # [owned, total]

for i, (player_id, _) in enumerate(xy_map):
    label = y_test[i]
    user_counts[player_id][0] += label
    user_counts[player_id][1] += 1

owned_percentages = [(owned / total) * 100 for owned, total in user_counts.values() if total > 0]
avg_owned_percent = np.mean(owned_percentages)

print(f"Average percent of owned games per user in test set: {avg_owned_percent:.2f}%")

genre_df = pd.read_csv('games_genres.csv')
genre_df['genres'] = genre_df['genres'].apply(ast.literal_eval)
game_id_to_genres = dict(zip(genre_df['gameid'], genre_df['genres']))


print("\n--- Example User Predictions ---")
shown = 0

for player_id, predictions in player_to_predictions.items():
    if shown >= 3:
        break

    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    top_n = sorted_preds[:N]
    actual_owned = set(game_id for game_id, _, label in predictions if label == 1)

    print(f"\nUser ID: {player_id}")
    print(f"Actual Owned Games in Test Set: {sorted(actual_owned)}")
    print(f"Top {N} Predicted Games:")

    for game_id, prob, _ in top_n:
        owned_flag = "yes" if game_id in actual_owned else "no"
        genres = game_id_to_genres.get(int(game_id), ["Unknown"])
        print(f"  Game ID: {game_id}, Prob: {prob:.4f}, Owned: {owned_flag}, Genres: {genres}")
    
    shown += 1
