import nflreadpy as nfl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load current season play-by-play data
pbp = nfl.load_pbp()

# Load player game-level stats for multiple seasons
player_stats = nfl.load_player_stats([2025])

print(player_stats)

# Load all available team level stats
team_stats = nfl.load_team_stats(seasons=True)

print(team_stats)

team_stats = nfl.load_team_stats([2021, 2022, 2023, 2024, 2025]).to_pandas()
schedules = nfl.load_schedules([2021, 2022, 2023, 2024, 2025]).to_pandas()
player_stats = nfl.load_player_stats([2021, 2022, 2023, 2024, 2025]).to_pandas()
ff_rankings = nfl.load_ff_rankings().to_pandas()

print("Team Stats")
team_stats.info()
print(team_stats.columns)
print("Schedules")
print(schedules.info())
# print("Player Stats")
# print(player_stats.info())
print("FF Rankings")
print(ff_rankings.info())



# print("-------------Team Stats------------------")
# print(team_stats.head())
# print("------------Schedules--------------------")
# print(schedules.head())
# print("-------------Player Stats----------------")
# print(player_stats.head())
# print("-----------------FF Rankings---------------")
# print(ff_rankings.head())

team_stats = team_stats[(team_stats["team"] == "SEA") | (team_stats["team"] == "NE")]
ff_rankings = ff_rankings[(ff_rankings["mergename"]=="Seattle Seahawks") | (ff_rankings["mergename"]=="New England Patriots")]

print(team_stats.head())
print(ff_rankings.head())
