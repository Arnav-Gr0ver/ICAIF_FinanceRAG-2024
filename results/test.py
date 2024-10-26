import pandas as pd

# Load the CSV file
csv_file = "combined_results.csv"  # Replace with the actual path to your CSV file
df = pd.read_csv(csv_file)

# Count occurrences of each query_id
query_counts = df['query_id'].value_counts()

# Find query_ids that appear more than 10 times
over_limit = query_counts[query_counts > 10]

# Check if any query_id has more than 10 occurrences
if not over_limit.empty:
    print("The following query_ids appear more than 10 times:")
    print(over_limit)
else:
    print("All query_ids appear 10 times or fewer.")
