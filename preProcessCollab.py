import pandas as pd              
import numpy as np 
import ast

steam_games_df = pd.read_csv("gaming_datasets/steam/games.csv")
# print(steam_games_df)
# print(steam_games_df.columns)
# print(10 in steam_games_df['gameid'].values)
# print(381210 in steam_games_df['gameid'].values)

players_and_their_games_df = pd.read_csv("gaming_datasets/steam/purchased_games.csv")
# print(players_and_their_games_df)
# print(players_and_their_games_df.columns)

games_to_remove = set()
games_to_keep = set()

for _, row in steam_games_df.iterrows():
    if pd.isna(row['genres']):
        games_to_remove.add(row['gameid']) # this adds all the games without generes to a hash set
    else:
        games_to_keep.add(row['gameid']) # adds all games with geners to another hash set

steam_games_df = steam_games_df[pd.notna(steam_games_df['genres'])] # removes the games with no generes
# print(3277430 in steam_games_df['gameid'].values) # game without genere
# print(3278740 in steam_games_df['gameid'].values) # game with genere




players_and_their_games_df = players_and_their_games_df[pd.notna(players_and_their_games_df['library'])] # removes players with no games

def remove_bad_games(game_list): # removes the unusable games from the player's libraries
    cleaned_list = []
    for game in game_list:
        if (game not in games_to_remove) and (game in games_to_keep):
            cleaned_list.append(game)

    return cleaned_list

# num_rows = 0

for idx, row in players_and_their_games_df.iterrows(): # does this for each player
    # if num_rows == 1:
    #     break
    player_games = row['library']
    player_games = ast.literal_eval(player_games) # turns the string array into an actual array
    # print(player_games)
    cleaned_games = remove_bad_games(player_games)
    players_and_their_games_df.at[idx, 'library'] = cleaned_games # replaces the players old game array with the new cleaned one containing only games with generes
    # num_rows += 1


players_and_their_games_df = players_and_their_games_df[players_and_their_games_df['library'].apply(
    lambda x: len(x) > 0
    )]
 # if any empty libraries after cleaning, remove those

steam_games_df = steam_games_df.sort_values(by="gameid") # sorts all the games and their info (like genre) by game id


players_and_their_games_df.to_csv('player_games.csv', index=False)
steam_games_df.to_csv('games_genres.csv', index=False)

