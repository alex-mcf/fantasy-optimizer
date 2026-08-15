def compute_projection(df):
    """Use realized season totals as the historical performance measure.

    This project currently evaluates past drafts; it does not predict a future
    season. Keeping the ``projection`` column name avoids breaking downstream
    scoring code while making the units consistent with draft-cost points.
    """
    df = df.copy()
    if "pts_ttl" not in df.columns:
        raise KeyError("Historical scoring requires a 'pts_ttl' column.")
    df["projection"] = df["pts_ttl"]
    return df
