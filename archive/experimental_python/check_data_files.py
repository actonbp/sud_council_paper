import pandas as pd
import glob

total_rows = 0
non_empty_rows = 0
filtered_rows = 0

print("Analyzing CSV files...\n")

for p in sorted(glob.glob('data/*_Focus_[Gg]roup*full*.csv')):
    df = pd.read_csv(p)
    total = len(df)
    
    # Check for non-empty Text
    non_empty = df['Text'].notna().sum()
    
    # Count moderator rows
    moderators = df[df['Speaker'].astype(str).str.match(r'^[A-Z]{2,3}$', na=False)].shape[0]
    
    # Apply both filters
    filtered = df[df['Text'].notna() & ~df['Speaker'].astype(str).str.match(r'^[A-Z]{2,3}$', na=False)]
    
    print(f'{p}:')
    print(f'  Total rows: {total}')
    print(f'  Non-empty Text: {non_empty}')
    print(f'  Moderator rows: {moderators}')
    print(f'  After both filters: {len(filtered)}')
    print()
    
    total_rows += total
    non_empty_rows += non_empty
    filtered_rows += len(filtered)

print(f'Summary:')
print(f'  Total rows across all files: {total_rows}')
print(f'  Total non-empty Text: {non_empty_rows}')
print(f'  Total after filtering: {filtered_rows}')

# Check if any files are missing the Status column
print("\nChecking columns in each file:")
for p in sorted(glob.glob('data/*_Focus_Group_full*.csv')):
    df = pd.read_csv(p)
    print(f'{p}: {list(df.columns)}')