import pandas as pd
import numpy as np
from config.league_config import league_settings

# Read in and simplify data
# Data is sourced from fantasypros.com

adp_2022 = pd.read_csv("FantasyOptimizer/data/2022/Pre_2022_ADP(HPPR).csv")
adp_2022 = adp_2022[["Rank","Player","Team","POS","AVG"]]
results_2022 = pd.read_csv("FantasyOptimizer/data/2022/Post_2022_Results(HPPR).csv")
results_2022 = results_2022[["Rank","Player","Pos","Team","AVG","TTL"]]
joined_2022 = pd.merge(adp_2022, results_2022, on='Player', how='inner', suffixes=('_pre', '_post'))

adp_2023 = pd.read_csv("FantasyOptimizer/data/2023/Pre_2023_ADP(HPPR).csv")
adp_2023 = adp_2023[["Rank","Player","Team", "POS", "AVG"]]
results_2023 = pd.read_csv("FantasyOptimizer/data/2023/Post_2023_Results(HPPR).csv")
results_2023 = results_2023[["Rank","Player","Pos","Team","AVG","TTL"]]
joined_2023 = pd.merge(adp_2023, results_2023, on='Player', how='inner', suffixes=('_pre', '_post'))


adp_2024 = pd.read_csv("FantasyOptimizer/data/2024/Pre_2024_ADP(HPPR).csv")
adp_2024 = adp_2024[["Rank","Player","Team", "POS", "AVG"]]
results_2024 = pd.read_csv("FantasyOptimizer/data/2024/Post_2024_Results(HPPR).csv")
results_2024 = results_2024[["Rank","Player","Pos","Team","AVG","TTL"]]
joined_2024 = pd.merge(adp_2024, results_2024, on='Player', how='inner', suffixes=('_pre', '_post'))

adp_2025 = pd.read_csv("FantasyOptimizer/data/2025/Pre_2025_ADP(HPPR).csv")
adp_2025 = adp_2025[["Rank","Player","Team", "POS", "AVG"]]

# Scoring logic 
# To calculate value, compare a players last 3 predraft APD to their final scoring respectively
# Also compare players draft to their final rank
# Later, consider injury data and team ranks?
# Gain/lose points in intervals depending on how much over or under they were
# Gain/lose points for scoring more than the average of the round they were drafted (and each round above)
# Gain/lose points for doing this multiple years (rookies should lose points)
# Return a dictionary with players mapped to their score
def calculate_player_value(joined_2022, joined_2023, joined_2024):
    
    league_size = league_settings["league_size"]

    joined_2022["Year"] = 2022
    joined_2023["Year"] = 2023
    joined_2024["Year"] = 2024

    all_years = pd.concat(joined_2022, joined_2023, joined_2024)

    # Compute player rounds with correct league size
    all_years["Round"] = (all_years["AVG_pre"] / league_size).apply(np.ceil).astype(int)

    # Find average points per round
    round_averages = all_years.groupby(["Year", "Round"])["TTL"].mean().to_dict()

    # Calculate the expected values for players considering where they were drafted
    all_years["Expected"] = all_years.apply(lambda row: round_averages.get((row["Year"], row["Round"]), 0), axis=1) # Gets averages and adds them to their player

    player_scores = {}
    players_grouped = all_years.groupby("Player")

    for player, group in players_grouped:
        total_score = 0
        success_year_counter = 0
        years_played = len(group) # New players with one good year will be more risky than established players

        for _, row in group.iterrows(): # Index not needed
            actual = row["TTL"]
            expected = row["Expected"]
            round_num = row["Round"]
            year = row["Year"]

            # Gain/lose points dependant on scores
            performance_modifier = (actual - expected) // 5

            round_modifier = 0
            max_round = all_years["Round"].max()

            for r in range(1, round_num):
                r_avg = round_averages.get((year, r), 0)
                if actual > r_avg:
                    round_modifier += 1

            for r in range(round_num + 1, max_round + 1):
                r_avg = round_averages.get((year, r), 0)
                if actual < r_avg:
                    round_bonus -= 1

            # TODO: potential durability modifier?

            # Track history
            if performance_modifier > 0:
                success_year_counter += 1

            total_score += performance_modifier + (round_modifier * 7) # TODO: durability
        
        # Penalize players who haven't played all 3 years
        sample_size_modifier = -7 if years_played == 1 else 0

        # Weight those who are consistent to have better scores
        consistency_modifier = {1: 1.0, 2: 1.5, 3: 2.0}.get(success_year_counter, 1.0)

        final_score = round((total_score + sample_size_modifier) * consistency_modifier, 2)
        player_scores[player] = final_score

    return player_scores

if __name__ == "__main__":
    # Run the scoring function
    player_scores = calculate_player_value(joined_2022, joined_2023, joined_2024)

    # Sort and print top 20 players
    top_players = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
    for player, score in top_players[:20]:
        print(f"{player.title():<25}  Score: {score}")