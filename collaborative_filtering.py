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
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def un_string_df_array(df, column_name):
    for idx, row in df.iterrows(): # does this for each player
        string_arr = row[column_name]
        arr = ast.literal_eval(string_arr) # turns the string array into an actual array
        df.at[idx, column_name] = arr # replaces said string array with said actual array
    
    return df


def remove_bad_games(game_list, games_to_keep): # removes the unusable games from the player's libraries
    cleaned_list = []
    for game in game_list:
        if (game in games_to_keep):
            cleaned_list.append(game) # creates a new list from the players old game list only containing games which are withing the top n games set created earlier

    return cleaned_list

def has_enough_games(games):
    return len(games) >= 10


def keep_top_n_games(num_games, player_games_df):
    all_game_ocur = []
    for idx, row in player_games_df.iterrows(): # does this for each player
        player_games = row['library']
        all_game_ocur.extend(player_games) # adds all the games for the current player to the list containing everyones games
    
    game_counts = Counter(all_game_ocur) # counter turns the big list with all games into a hashmap with counted occurences of each game

    top_games = game_counts.most_common(num_games)  # top n most played (list of tuples (game_id, count))

    # print(top_games)

    top_game_ids_set = set()
    top_game_ids_arr = []

    for top in top_games:
        top_game_ids_set.add(top[0]) # adds the top n games (their ids) to a set
        top_game_ids_arr.append(top[0]) # adds the top n games (their ids) to an array

    for idx, row in player_games_df.iterrows(): # remove non top games from player libraries
        player_games = row['library']
        cleaned_games = remove_bad_games(player_games, top_game_ids_set)
        player_games_df.at[idx, 'library'] = cleaned_games # replaces the players old list of games with the new cleaned one only containing games withing the top n games

    player_games_df = player_games_df[player_games_df['library'].apply(has_enough_games)] # keeps players with a library of above a certain amount of games (function currently checks if they have 10 or more games as they did in the paper)
    
    player_games_df.to_csv(f'player_games_{num_games}.csv', index=False) # creates a new csv with the players and their new game libraries only containing games within the top n games

    np.save(f'top_{num_games}_arr.npy', top_game_ids_arr)

def encode_user_arr(user_game_list, top_games, top_games_map):
    encoded_arr = [0] * len(top_games)
    for game_id in user_game_list:
        if game_id in top_games_map:
            encoded_arr[top_games_map[game_id]] = 1

    return encoded_arr

# model arcitecure is based off (Using Deep Learning and Steam User Data for Beer Video Game Recommendations) paper

def create_autoencoder(input_dim):

    input_layer = Input(shape=(input_dim,)) # creates the input layer for our model that takes in the vector

    # encoder part
    # hidden layer 1:
    # 256 neurons, ReLU activate
    x = Dense(256, activation='relu')(input_layer)

    # hidden layer 2:
    # 128 neurons, ReLU activate
    x = Dense(128, activation='relu')(x)

    # decoder part
    # hidden layer 3:
    # 256 neurons, ReLU activate
    x = Dense(256, activation='relu')(x)

    # output layer:
    # sigmoid activate
    output_layer = Dense(input_dim, activation='sigmoid')(x)

    # create model object
    collab_model = Model(inputs=input_layer, outputs=output_layer)

    # model "was trained using the mean absolute error loss function, the Adam optimizer with a learning rate of 0.001, and for 20 training epochs."

    # set up params for backpropagation and weight updates.
    # collab_model.compile(
    #     optimizer=Adam(learning_rate=0.001),
    #     loss='mae',
    #     metrics=['mae'] # accuracy, mae or mse
    #     # used to calculate and display the mean absolute error for each batch and each epoch (should differ from loss if using validation data)
    # )

    collab_model.compile( # gives better results
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(),
        metrics=['binary_accuracy']
    )

    # Print model summary
    collab_model.summary()

    return collab_model

def mask_data(input_data, percent_masked=0.2):
    masked_data = input_data.copy()

    # This will store which indices were masked for each user (optional, useful for evaluation)
    mask_indices = []

    # Loop through each user's game vector
    for i in range(len(masked_data)):

        current_user_vec = masked_data[i]

        # Find all indices where the user owns a game (i.e., where the value is 1)
        owned_indices = np.where(current_user_vec == 1)[0]
        # np.where() returns the indices of True values in the array.

        # Determine how many of those should be masked (based on percent)
        num_to_mask = int(len(owned_indices) * percent_masked)

        # If the user doesn't have enough owned games to mask, skip
        if num_to_mask == 0:
            mask_indices.append([])
            continue

        # Randomly select a subset of the owned indices to be masked (set to 0)
        masked = np.random.choice(owned_indices, size=num_to_mask, replace=False)

        # Set those selected positions to 0 in the input vector (simulate the user "not owning" them)
        masked_data[i, masked] = 0

        # Save the masked indices for this user (for evaluation later)
        mask_indices.append(masked)

    # Return:
    # - masked_data: input vectors with some 1s masked to 0
    # - input_data: original (unmasked) target vectors
    # - mask_indices: which 1s were masked for each user
    return masked_data, input_data, mask_indices




def main():

    # code to shorten considered games
    # player_games_df = pd.read_csv('player_games.csv')
    # player_games_df = un_string_df_array(player_games_df, 'library')
    # keep_top_n_games(1000, player_games_df)

    player_games_df = pd.read_csv('player_games_1000.csv')
    player_games_df = un_string_df_array(player_games_df, 'library')
    sampled_df = player_games_df.sample(n=40000, random_state=42) # gets a random sample of 40000 users


    top_1000_arr = np.load('top_1000_arr.npy', allow_pickle=True)
    top_1000_arr = top_1000_arr.tolist()
    # print(top_1000_arr)


    # gives each game their index in the numpy array so you check which game is is which position
    game_ids_pos = {}
    for i in range(len(top_1000_arr)):
        game_ids_pos[top_1000_arr[i]] = i

    user_ids = [] # A list to keep track of which user corresponds to which row in final matrix 
    input_data = [] # A list that will eventually become a 2D NumPy array containing all the binary vectors

    for _, row in sampled_df.iterrows():
        user_ids.append(row['playerid'])
        encoded = encode_user_arr(row['library'], top_1000_arr, game_ids_pos)
        input_data.append(encoded)

    input_data = np.array(input_data) # converts to numpy array for model

    # print(game_ids_pos)
    # print(user_ids)
    # print(input_data)

    X_data, y_data, masked_indices = mask_data(input_data)


    # Split both masked inputs (X) and original targets (y), along with user IDs
    X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = train_test_split(
        X_data, y_data, user_ids, test_size=0.2, random_state=42
    )

    input_dim = X_data.shape[1] # should be 1000 since thats number of games
    # change later to num of games/features (each row is a user and each column is a game)

    collab_model = create_autoencoder(input_dim)

    # Evaluate untrained model
    print("\n[Pre-Training Evaluation]")
    loss, mae = collab_model.evaluate(X_test, y_test, verbose=0)
    # print(f"Untrained MAE: {mae:.4f}")

    predictions = collab_model.predict(X_test)

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

    print(f"pre_Accuracy: {accuracy:.4f}")
    print(f"pre_Precision: {precision:.4f}")
    print(f"pre_Recall:    {recall:.4f}")
    print(f"pre_F1 Score:  {f1:.4f}")


    # set up train params/run training of model
    collab_model.fit(
        X_train, y_train, # since we want what the auto encoder is recreating the same structure as the input, both trains should be the same
        epochs=20,
        batch_size=64,
        shuffle=True,
        validation_split=0.1
    )

    # run model on test set
    predictions = collab_model.predict(X_test)

    # save model
    collab_model.save('collab_model3_20epoch.h5')

    # Save test data
    np.save('collab_data/X_test3.npy', X_test)
    np.save('collab_data/y_test3.npy', y_test)
    np.save('collab_data/user_ids_test3.npy', np.array(user_ids_test))

    # testing code:
    model = load_model('collab_model3_20epoch.h5')

    loss, mae = model.evaluate(X_test, y_test)

    # print(f"MAE: {mae}")

    predictions = model.predict(X_test)  # shape: (num_users, 1000)

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





    




if __name__ == "__main__":
    main()

