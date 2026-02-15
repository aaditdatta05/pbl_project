import pandas as pd

def main(csv_path="data/train.csv"):
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns:
        raise ValueError("CSV must include a 'diagnosis' column.")

    counts = df["diagnosis"].value_counts().sort_index()
    total = int(counts.sum())

    print("Class counts:")
    for label, count in counts.items():
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {int(label)}: {int(count)} ({pct:.2f}%)")

    print(f"Total samples: {total}")

if __name__ == "__main__":
    main()
