import random

# Read lines from the first file
with open('/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/seetopic_data/scidocs.txt', 'r') as file1:
    lines1 = file1.readlines()

# Read lines from the second file
with open('/home/chris/COMP4951-Thesis-Out-of-Vocab-Seed-Mining/src/data/seetopic_data/amazon.txt', 'r') as file2:
    lines2 = file2.readlines()

# Combine the lines from both files
combined_lines = lines1 + lines2

# Shuffle the combined lines
random.shuffle(combined_lines)

# Print or do something with the shuffled lines
with open('data/seetopic_data/merged.txt', 'w') as new_file:
    for line in combined_lines:
        new_file.write(line)
