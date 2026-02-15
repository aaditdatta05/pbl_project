import pandas as pd

df = pd.read_csv("data/test.csv", dtype=str)

# Keep only rows where image_id is a clean hexadecimal string
clean_df = df[~df["image_id"].str.contains("e", case=False, na=False)]

print(f"Original rows: {len(df)}")
print(f"Clean rows: {len(clean_df)}")
print(f"Deleted rows: {len(df) - len(clean_df)}")

clean_df.to_csv("data/test_clean.csv", index=False)

print("✓ Saved cleaned file as data/test_clean.csv")