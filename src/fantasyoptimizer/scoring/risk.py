def compute_risk(df):
    """Estimate multi-season inconsistency for each player.

    Risk is the population standard deviation of points per game across the
    selected seasons. It is converted to season-scale points before applying
    the penalty so its units match the rest of the score.
    """
    df = df.copy()
    if "pts_avg" not in df.columns:
        raise KeyError("Risk scoring requires a 'pts_avg' column.")
    df["ppg"] = df["pts_avg"]
    player_column = "player_key" if "player_key" in df.columns else "player"
    df["risk_sample_size"] = df.groupby(player_column)["ppg"].transform("count")
    df["risk"] = df.groupby(player_column)["ppg"].transform(
        lambda values: values.std(ddof=0)
    )
    df.loc[df["risk_sample_size"] < 2, "risk"] = float("nan")
    df["risk_penalty"] = df["risk"].fillna(0) * 17 * 0.3
    return df
