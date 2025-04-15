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


def un_string_df_array(df, column_name):
    for idx, row in df.iterrows(): # does this for each player
        string_arr = row[column_name]
        arr = ast.literal_eval(string_arr)
        df.at[idx, column_name] = arr
    
    return df


def remove_bad_games(game_list, games_to_keep): # removes the unusable games from the player's libraries
    cleaned_list = []
    for game in game_list:
        if (game in games_to_keep):
            cleaned_list.append(game)

    return cleaned_list

def has_enough_games(games):
    return len(games) >= 10


def keep_top_n_games(num_games, player_games_df):
    all_game_ocur = []
    for idx, row in player_games_df.iterrows(): # does this for each player
        player_games = row['library']
        all_game_ocur.extend(player_games)
    
    game_counts = Counter(all_game_ocur) # counter turns the big list with all games into a hashmap with counted occurences of each game

    top_games = game_counts.most_common(num_games)  # top 1000 most played (list of tuples (game_id, count))

    # print(top_games)

    top_game_ids_set = set()
    top_game_ids_arr = []

    for tup in top_games:
        top_game_ids_set.add(tup[0])
        top_game_ids_arr.append(tup[0])

    for idx, row in player_games_df.iterrows(): # remove non top games from player libraries
        player_games = row['library']
        cleaned_games = remove_bad_games(player_games, top_game_ids_set)
        player_games_df.at[idx, 'library'] = cleaned_games

    player_games_df = player_games_df[player_games_df['library'].apply(has_enough_games)] # keeps players with a library of above a certain amount of games
    
    player_games_df.to_csv(f'player_games_{num_games}.csv', index=False)

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
    collab_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mae',
        metrics=['mae'] # accuracy, mae or mse
        # used to calculate and display the mean absolute error for each batch and each epoch (should differ from loss if using validation data)
    )

    # Print model summary
    collab_model.summary()

    return collab_model


def main():

    # code to shorten considered games
    # player_games_df = pd.read_csv('player_games.csv')
    # player_games_df = un_string_df_array(player_games_df, 'library')
    # keep_top_n_games(1000, player_games_df)

    player_games_df = pd.read_csv('player_games_1000.csv')
    player_games_df = un_string_df_array(player_games_df, 'library')
    sampled_df = player_games_df.sample(n=40000, random_state=42) # gets a random sample of 500 users


    top_1000_arr = np.load('top_1000_arr.npy', allow_pickle=True)
    top_1000_arr = top_1000_arr.tolist()
    print(top_1000_arr)


    game_ids_pos = {}
    for i in range(len(top_1000_arr)):
        game_ids_pos[top_1000_arr[i]] = i

    user_ids = [] # A list to keep track of which user corresponds to which row in final matrix 
    X_data = [] # A list that will eventually become a 2D NumPy array containing all the binary vectors

    for _, row in sampled_df.iterrows():
        user_ids.append(row['playerid'])
        encoded = encode_user_arr(row['library'], top_1000_arr, game_ids_pos)
        X_data.append(encoded)

    X_data = np.array(X_data) # converts to numpy array for model

    # set up test and train data
    X_train, X_test, user_ids_train, user_ids_test = train_test_split(
        X_data, user_ids, test_size=0.2, random_state=42
        )

    input_dim = X_data.shape[1] # should be 1000 since thats number of games
    # change later to num of games/features (each row is a user and each column is a game)

    collab_model = create_autoencoder(input_dim)

    # set up train params/run training of model
    collab_model.fit(
        X_train, X_train, # since we want what the auto encoder is recreating the same structure as the input, both trains should be the same
        epochs=20,
        batch_size=128,
        shuffle=True,
        validation_split=0.1
    )

    # run model on test set
    predictions = collab_model.predict(X_test)

    # save model
    collab_model.save('collab_model1.h5')

    # testing code:
    model = load_model('collab_model1.h5')

    loss, mae = model.evaluate(X_test, X_test)
    print(f"MAE: {mae}")

    predictions = model.predict(X_test)  # shape: (num_users, 1000)

    print(predictions)




if __name__ == "__main__":
    main()

