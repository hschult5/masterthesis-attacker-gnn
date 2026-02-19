import os
import pandas as pd

def average_csvs_over_seed(
    folder,
    output_folder=None,
    seed_col="seed",
    suffix="_avg"
):
    """
    For every CSV in `folder`:
    - average numeric columns over the seed dimension
    - drop the seed column
    - preserve grouping by all non-numeric columns except seed
    - write a new CSV with `suffix` appended to the filename
    """

    if output_folder is None:
        output_folder = folder

    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(folder):
        if not fname.endswith(".csv"):
            continue

        path = os.path.join(folder, fname)
        df = pd.read_csv(path)

        if seed_col not in df.columns:
            print(f"[skip] {fname}: no '{seed_col}' column")
            continue

        # Identify columns
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = df.columns.difference(numeric_cols).tolist()

        # Remove seed from both
        numeric_cols = [c for c in numeric_cols if c != seed_col]
        group_cols = [c for c in non_numeric_cols if c != seed_col]

        # Group and average
        df_avg = (
            df
            .groupby(group_cols, as_index=False)[numeric_cols]
            .mean()
        )

        # Write output
        out_name = fname.replace(".csv", f"{suffix}.csv")
        out_path = os.path.join(output_folder, out_name)
        df_avg.to_csv(out_path, index=False)

        print(f"[ok] {fname} → {out_name}")