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
    df["risk"] = df.groupby("player")["ppg"].transform(
        lambda values: values.std(ddof=0)
    )
    df["risk_penalty"] = df["risk"] * 17 * 0.3
    return df
