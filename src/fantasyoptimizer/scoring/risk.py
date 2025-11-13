# risk.py

# TODO: account for seasons played, injury history, and consistency
def compute_risk(df):
    df = df.copy()

    if "pts_ttl" in df.columns:
        df["games"] = 17  # adjust if you track real games
        df["ppg"] = df["pts_ttl"] / df["games"]
        df["risk"] = df["ppg"].rolling(3, min_periods=1).std().fillna(0)
    else:
        df["risk"] = 0

    # Higher risk = penalize score
    df["risk_penalty"] = df["risk"] * 0.3

    return df