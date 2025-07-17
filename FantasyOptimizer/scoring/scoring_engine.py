# Scoring logic 
# What makes a player have a good grade? 
# 1. Calculate value
# 2. Compare value to previous ADP
# 3. Project value to upcoming Mock ADP

# To calculate value, compare a players last 3 predraft APD to their final scoring respectively
# Later, consider injury data and team ranks?

import pandas as pd

# Read in and simplify data
# Data is sourced from fantasypros.com

adp_2022 = pd.read_csv("FantasyOptimizer/data/2022/Pre_2022_ADP(HPPR).csv")
adp_2022 = adp_2022[["Rank","Player","Team","POS","AVG"]]
results_2022 = pd.read_csv("FantasyOptimizer/data/2022/Post_2022_Results(HPPR).csv")
results_2022 = results_2022[["Rank","Player","Pos","Team","AVG","TTL"]]

adp_2023 = pd.read_csv("FantasyOptimizer/data/2023/Pre_2023_ADP(HPPR).csv")
adp_2023 = adp_2023[["Rank","Player","Team", "POS", "AVG"]]
results_2023 = pd.read_csv("FantasyOptimizer/data/2023/Post_2023_Results(HPPR).csv")
results_2023 = results_2023[["Rank","Player","Pos","Team","AVG","TTL"]]

adp_2024 = pd.read_csv("FantasyOptimizer/data/2024/Pre_2024_ADP(HPPR).csv")
adp_2024 = adp_2024[["Rank","Player","Team", "POS", "AVG"]]
results_2024 = pd.read_csv("FantasyOptimizer/data/2024/Post_2024_Results(HPPR).csv")
results_2024 = results_2024[["Rank","Player","Pos","Team","AVG","TTL"]]

adp_2025 = pd.read_csv("FantasyOptimizer/data/2025/Pre_2025_ADP(HPPR).csv")
adp_2025 = adp_2025[["Rank","Player","Team", "POS", "AVG"]]

