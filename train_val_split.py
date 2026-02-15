import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/train.csv")
df = df.rename(columns={"diagnosis": "label"})
df = df.rename(columns={"id_code": "image_id",})

if "image_id" not in df.columns:
    raise ValueError("train.csv has no column named image_id — send me your CSV header.")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val.csv", index=False)

print("Split complete")
print("Train size:", len(train_df))
print("Val size:", len(val_df))