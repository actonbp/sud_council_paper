import pandas as pd
import glob
import re

# Collect all unique speakers
all_speakers = set()
speaker_counts = {}

for p in glob.glob('data/*_Focus_Group_full*.csv'):
    df = pd.read_csv(p)
    speakers = df['Speaker'].dropna().astype(str).unique()
    all_speakers.update(speakers)
    
    # Count utterances per speaker type
    for s in speakers:
        if s not in speaker_counts:
            speaker_counts[s] = 0
        speaker_counts[s] += len(df[df['Speaker'].astype(str) == s])

# Categorize speakers
two_letter = []
three_letter = []
numbers = []
other = []

for s in sorted(all_speakers):
    if re.match(r'^[A-Z]{2}$', s):
        two_letter.append(s)
    elif re.match(r'^[A-Z]{3}$', s):
        three_letter.append(s)
    elif s.isdigit():
        numbers.append(s)
    else:
        other.append(s)

print('Speaker Analysis:')
print(f'2-letter speakers ({len(two_letter)}): {two_letter}')
print(f'3-letter speakers ({len(three_letter)}): {three_letter}')
print(f'Numeric speakers ({len(numbers)}): {sorted(numbers, key=int)[:20]}{"..." if len(numbers) > 20 else ""}')
print(f'Other speakers ({len(other)}): {other}')

# Show utterance counts for each type
print('\nUtterance counts by speaker type:')
two_three_count = sum(speaker_counts[s] for s in speaker_counts if re.match(r'^[A-Z]{2,3}$', s))
numeric_count = sum(speaker_counts[s] for s in speaker_counts if s.isdigit())
other_count = sum(speaker_counts[s] for s in speaker_counts if s not in two_letter + three_letter + numbers)

print(f'2-3 letter speakers (moderators): {two_three_count} utterances')
print(f'Numeric speakers (participants): {numeric_count} utterances')
print(f'Other speakers: {other_count} utterances')