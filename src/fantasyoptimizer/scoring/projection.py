# projection.py

# TODO: implement more advanced projection logic (e.g., regression, machine learning, etc.)
def compute_projection(df):
    df = df.copy()

    # Current placeholder: use last year’s pts_avg
    df["projection"] = df["pts_avg"]

    return df