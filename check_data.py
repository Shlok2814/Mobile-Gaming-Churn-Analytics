import os
import pandas as pd

def check_dataset(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Please ensure 'cookie_cats.csv' is placed inside the '/data' folder.")
        return

    print("=" * 50)
    print(f"Loading Dataset: {filepath}")
    print("=" * 50)

    df = pd.read_csv(filepath)

    print("\n--- DATASET INFO ---")
    df.info()

    print("\n--- FIRST 10 ROWS ---")
    print(df.head(10))

    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe(include='all'))

    print("\n--- NULL VALUES CHECK ---")
    null_counts = df.isnull().sum()
    print(null_counts)
    if null_counts.sum() == 0:
        print("RESULT: No missing values found!")
    else:
        print(f"RESULT: Found {null_counts.sum()} total missing values.")

    print("\n--- DUPLICATE USERID CHECK ---")
    if 'userid' in df.columns:
        duplicate_count = df['userid'].duplicated().sum()
        print(f"Duplicate userids: {duplicate_count}")
        if duplicate_count == 0:
            print("RESULT: All user IDs are unique!")
        else:
            print(f"WARNING: Found {duplicate_count} duplicate user IDs.")
    else:
        print("Column 'userid' not found in dataset.")

    print("\n=" * 50)
    print("Check completed.")
    print("=" * 50)

if __name__ == "__main__":
    data_path = os.path.join("data", "cookie_cats.csv")
    check_dataset(data_path)
