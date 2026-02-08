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

# Data is originally in polars form
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

# print(team_stats.head())
# print(ff_rankings.head())
# print(schedules["season"])

# print(team_stats.iloc[:, 0:10].head)
# print(team_stats.iloc[:, 10:20].head)
# print(team_stats.iloc[:, 20:30].head)
# print(team_stats.iloc[:, 30:40].head)
# print(team_stats.iloc[:, 40:50].head)
# print(team_stats.iloc[:, 50:60].head)
# print(team_stats.iloc[:, 60:70].head)
# print(team_stats.iloc[:, 70:80].head)
# print(team_stats.iloc[:, 80:90].head)
# print(team_stats.iloc[:, 90:100].head)
# print(team_stats.iloc[:, 100:110].head)


#team_stats = nfl.load_team_stats([2021, 2022, 2023, 2024, 2025]).to_pandas()

# ---- Build season-level stats per team ----
season_stats = team_stats.groupby(["season", "team"]).agg(
    pass_yards=("passing_yards", "sum"),
    rush_yards=("rushing_yards", "sum"),
    pass_epa=("passing_epa", "sum"),
    rush_epa=("rushing_epa", "sum"),
    sacks=("def_sacks", "sum"),
    ints=("def_interceptions", "sum")
).reset_index()

season_stats["total_yards"] = season_stats["pass_yards"] + season_stats["rush_yards"]
season_stats["total_epa"] = season_stats["pass_epa"] + season_stats["rush_epa"]
season_stats["defense"] = season_stats["sacks"] + season_stats["ints"]

# ---- Create target variable in games (home win) ----
games = schedules.copy()
games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

# ----  Merge season stats for home team ----
games = games.merge(
    season_stats[["season", "team", "total_yards", "total_epa", "defense"]],
    left_on=["season", "home_team"],
    right_on=["season", "team"],
    how="left"
)
games = games.rename(columns={
    "total_yards": "home_yards",
    "total_epa": "home_epa",
    "defense": "home_defense"
}).drop(columns=["team"])

# ---- Merge season stats for away team ----
games = games.merge(
    season_stats[["season", "team", "total_yards", "total_epa", "defense"]],
    left_on=["season", "away_team"],
    right_on=["season", "team"],
    how="left"
)
games = games.rename(columns={
    "total_yards": "away_yards",
    "total_epa": "away_epa",
    "defense": "away_defense"
}).drop(columns=["team"])

# ---- Create difference features ----
games["yards_diff"] = games["home_yards"] - games["away_yards"]
games["epa_diff"] = games["home_epa"] - games["away_epa"]
games["defense_diff"] = games["home_defense"] - games["away_defense"]
games["home_field"] = 1  # not needed for neutral site, will override later

features = ["yards_diff", "epa_diff", "defense_diff", "home_field"]
X = games[features]
y = games["home_win"]

# ----Train Random Forest ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# ---- Super Bowl prediction for NE vs SEA ----
ne_stats = season_stats[(season_stats["season"] == 2025) & (season_stats["team"] == "NE")].iloc[0]
sea_stats = season_stats[(season_stats["season"] == 2025) & (season_stats["team"] == "SEA")].iloc[0]

superbowl_features = pd.DataFrame({
    "yards_diff": [ne_stats["total_yards"] - sea_stats["total_yards"]],
    "epa_diff": [ne_stats["total_epa"] - sea_stats["total_epa"]],
    "defense_diff": [ne_stats["defense"] - sea_stats["defense"]],
    "home_field": [0]  # neutral site
})

proba = model.predict_proba(superbowl_features)[0][1]
percentage= round(proba, 2) *100
print("Predicted New England Patriots win probability vs Seattle Seahawks:", percentage, "%")
