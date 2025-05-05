import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import ast
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# Load model and test data
model = load_model('collab_model3_20epoch.h5')
X_test = np.load('collab_data/X_test3.npy')
y_test = np.load('collab_data/y_test3.npy')

# Reconstruct masked indices
masked_indices = []
for x_vec, y_vec in zip(X_test, y_test):
    masked = np.where((y_vec == 1) & (x_vec == 0))[0]
    masked_indices.append(masked)

predictions = model.predict(X_test)

# --- MAE Evaluation ---
loss, mae = model.evaluate(X_test, y_test, verbose=0)
# print(f"MAE: {mae:.4f}")

# --- Binary Classification Evaluation ---
# Convert probabilities to binary predictions
y_pred_binary = (predictions >= 0.5).astype(int)

# Flatten arrays for sklearn metrics
y_true_flat = y_test.flatten()
y_pred_flat = y_pred_binary.flatten()

# Compute metrics
precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
accuracy = accuracy_score(y_true_flat, y_pred_flat)  

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


top_k = 5
hits = 0
total_users = 0

for i, user_pred in enumerate(predictions):
    true_masked = masked_indices[i]
    if len(true_masked) == 0:
        continue

    # Get all unowned/unseen games
    unowned = np.where((y_test[i] == 0) & (X_test[i] == 0))[0]
    if len(unowned) < len(true_masked):
        continue  # skip if not enough negatives

    # Sample equal number of unowned games
    sampled_unowned = np.random.choice(unowned, size=len(true_masked), replace=False)

    # Combine indices
    eval_indices = np.concatenate([true_masked, sampled_unowned])

    rng = np.random.default_rng(seed=42)  # consistent random shuffling
    rng.shuffle(eval_indices)

    # Get predicted scores for selected games
    scored_games = [(idx, user_pred[idx]) for idx in eval_indices]

    # Sort by predicted score, get top-K
    top_indices = [idx for idx, _ in sorted(scored_games, key=lambda x: x[1], reverse=True)[:top_k]]

    # Count how many of top-K are actually masked (true positives)
    hit_count = len(set(top_indices) & set(true_masked))
    hits += hit_count
    total_users += 1

hit_rate = hits / (total_users * top_k) if total_users > 0 else 0
print(f"Adjusted Hit Rate @ {top_k}: {hit_rate:.4f}")

genre_df = pd.read_csv('games_genres.csv')
genre_df['genres'] = genre_df['genres'].apply(ast.literal_eval)
game_id_to_genres = dict(zip(genre_df['gameid'], genre_df['genres']))

top_1000_games = np.load('top_1000_arr.npy', allow_pickle=True)
user_ids = np.load('collab_data/user_ids_test2.npy', allow_pickle=True)

# Show 3 example users
print("\n--- Example User Predictions ---")
shown = 0

for i, user_pred in enumerate(predictions):
    true_masked = masked_indices[i]
    if len(true_masked) < top_k:
        continue

    unowned = np.where((y_test[i] == 0) & (X_test[i] == 0))[0]
    if len(unowned) < len(true_masked):
        continue

    sampled_unowned = np.random.choice(unowned, size=len(true_masked), replace=False)
    eval_indices = np.concatenate([true_masked, sampled_unowned])

    scored = [(idx, user_pred[idx], idx in true_masked) for idx in eval_indices]
    top_preds = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    print(f"\nUser ID: {user_ids[i]}")
    print(f"Actual Masked (Owned) Games: {[int(top_1000_games[idx]) for idx in sorted(true_masked)]}")
    print(f"Top-{top_k} Predictions:")

    for idx, score, is_masked in top_preds:
        game_id = int(top_1000_games[idx])
        genres = game_id_to_genres.get(game_id, ["Unknown"])
        print(f"  Game ID: {game_id}, Score: {score:.4f}, Masked Owned: {'yes' if is_masked else 'no'}, Genres: {genres}")

    shown += 1
    if shown >= 3:
        break
