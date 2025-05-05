from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
# The DAE was trained and tested using Keras and Google’s TensorFlow library.
import pandas as pd              
import numpy as np 
import ast
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import tensorflow as tf
import os
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'


print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {gpus[0].name}")
    try:
        # Prevent TensorFlow from allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print("Could not set memory growth:", e)
else:
    print("No GPU detected. Training will use CPU.")


def un_string_df_array(df, column_name):
    for idx, row in df.iterrows(): # does this for each player
        string_arr = row[column_name]
        arr = ast.literal_eval(string_arr) # turns the string array into an actual array
        df.at[idx, column_name] = arr # replaces said string array with said actual array
    
    return df

game_info_df = pd.read_csv('games_genres.csv')

genre_df = game_info_df[['gameid','developers', 'genres']]

genre_df = un_string_df_array(genre_df, 'genres')

player_games_df = pd.read_csv('player_games_1000.csv')


mlb = MultiLabelBinarizer() # tool used to convert the lists of genres/devlopers into a numerical representation

genre_matrix = mlb.fit_transform(genre_df['genres']) # assigns each game a list where each index represents a genre and has a 1 if the game is said genre

encoded_genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

genre_df = pd.concat([genre_df, encoded_genre_df], axis=1) # adds the encoded games to the genre df

# print(mlb.classes_)

# print(encoded_genre_df)

# print(genre_matrix)

# print(genre_df)

game_id_to_vector = dict(zip(genre_df['gameid'], genre_matrix)) # creates a dictionary that maps each game id to its repective genre vector

# print(game_id_to_vector)

player_games_df = un_string_df_array(player_games_df, 'library')

# print(player_games_df)

player_vectors = {}

left_out_player_games = {}

percent_masked = 0.5

for idx, row in player_games_df.iterrows():
    player_id = row['playerid']
    player_games = row['library']
    player_vectors[player_id] = np.zeros(len(mlb.classes_))

    num_to_mask = int(len(player_games) * percent_masked)

    masked_set = set(np.random.choice(player_games, size=num_to_mask, replace=False)) # creates a set of game_ids that are not used for creating the player genre vectors 
    
    num_not_left_out = 0

    for i in range(len(player_games)):
        game_id = player_games[i]

        if (game_id not in masked_set):
            player_vectors[player_id] += game_id_to_vector[game_id]
            num_not_left_out += 1
        else:
            if player_id in left_out_player_games: # creates a dic using the player_id: [left_out_games]
                left_out_player_games[player_id].append(game_id)
            else:
                left_out_player_games[player_id] = [game_id]
    
    if num_not_left_out > 0:
        player_vectors[player_id] /= num_not_left_out # from the players past games gets the avg of the genres they played

print("\ndone creating vectors\n")
# print(player_vectors)

# code below is to create an even split of positive and negative samples for the datasets

positive_samples = []
negative_samples = []

# pos_test_samples = []
# neg_test_samples = []


all_game_ids = set(genre_df['gameid'])  # Create a set of all known games from your genres database.

for player_id, p_vector in player_vectors.items():  # Loop through every player and their genre preference vector.

    current_player_games = player_games_df[player_games_df['playerid'] == player_id] # creates a df with 1 row that is for the current player_id and their games

    owned_games = set(current_player_games['library'].iloc[0]) # creates a set out of the owned games

    not_owned_games = list(all_game_ids - owned_games) # creates a list of all the games that the player does not own

    left_out_games = set(left_out_player_games[player_id])

    num_masked_games = 0

    # Positive examples (owns)
    for game_id in owned_games:  # Loop through all games the player owns
        if game_id in game_id_to_vector:  # Make sure the game has a genre vector
            if (game_id in left_out_games): # only creates samples from games that were not used to create the player genre vectors
                g_vector = game_id_to_vector[game_id]  # Get the genre vector for the current game
                input_vector = np.concatenate([p_vector, g_vector])  # Combine player and game vectors into one input vector
                positive_samples.append((input_vector, 1, (player_id, game_id))) # Add a tuple: (input vector, label=1 for 'player owns game', (plaer_id, game_id))
                num_masked_games += 1

    # Negative examples (does not own)
    sampled_negatives = random.sample(not_owned_games, min(len(owned_games), len(not_owned_games))) # since the num games that the player does not own vastly out weighs the num that they do, we randomlly get an equal num of unknown games for the player as known to loop throough

    neg_test_ids = set(np.random.choice(sampled_negatives, size=num_masked_games, replace=False)) # makes it equal to the amount of masked ostive samples

    for game_id in sampled_negatives:  # Same process, but for negative (unowned) examples
        if game_id in game_id_to_vector:
            if (game_id in neg_test_ids):
                g_vector = game_id_to_vector[game_id]
                input_vector = np.concatenate([p_vector, g_vector])
                negative_samples.append((input_vector, 0, (player_id, game_id)))  # Label = 0 means the player does not own this game

# Combine and shuffle
all_samples = positive_samples + negative_samples
random.Random(42).shuffle(all_samples)

# test_samples = pos_test_samples + neg_test_samples
# random.Random(42).shuffle(test_samples)

print("\ndone creating pos and neg samples\n")

# Separate inputs and labels

X = np.array([x for x, _, _ in all_samples]) #  Extracts all input vectors from the (input_vector, label) tuples and creates a 2D NumPy array of shape (num_samples, input_vector_length)

# X_train = X

y = np.array([label for _, label, _ in all_samples])  # Extracts all labels from the (input_vector, label) tuples and creates a 1D NumPy array of shape (num_samples,)

# y_train = y

xy_map = np.array([meta for _, _, meta in all_samples])  # (player_id, game_id) tuples



# X_test = np.array([x for x, _, _ in test_samples]) #  Extracts all input vectors from the (input_vector, label) tuples and creates a 2D NumPy array of shape (num_samples, input_vector_length)

# y_test = np.array([label for _, label, _ in test_samples])  # Extracts all labels from the (input_vector, label) tuples and creates a 1D NumPy array of shape (num_samples,)

# xy_map_test = np.array([meta for _, _, meta in test_samples])  # (player_id, game_id) tuples

print("\ndone creating inputs and outputs\n")

# create model

input_size = X.shape[1]  # input vector size (player + game vector)

model = Sequential([  # Each layer narrows the feature space to focus on the most important patterns and ultimately compresses it to a single output
    Dense(128, input_dim=input_size, activation='relu'),  # First hidden layer with ReLU activation for non-linearity
    Dropout(0.2),  # During training, randomly sets 20% of neuron outputs to 0
                   # Helps prevent overfitting by forcing the model to learn more robust and distributed patterns
    Dense(64, activation='relu'),  # Second hidden layer continues extracting compressed, meaningful features
    Dense(1, activation='sigmoid')  # Output layer: returns a probability (0 to 1) that the player owns/likes the game
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test, xy_map_train, xy_map_test = train_test_split(X, y, xy_map, test_size=0.2, random_state=42)

print("\ndone creating model arc and splitting data\n")

print("\npre train eval:\n")
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

print("\ndone with pre eval\n")

# Then train on X_train and y_train
model.fit(X_train, y_train, epochs=10, batch_size=128, shuffle=True)
# model.fit(X_train, y_train, epochs=2, batch_size=128, shuffle=True) # faster

print("\ndone training\n")

# Save model/test data after training
model.save('content_model3_10epoch.h5')

np.save('content_model_data/X_test3.npy', X_test)
np.save('content_model_data/y_test3.npy', y_test)
np.save('content_model_data/xy_map3.npy', xy_map_test)



with open('content_model_data/mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)