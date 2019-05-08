# PROCEDURE

# 1. CSV all the data
# 2. Make pd dataframe
# 3. Make all data ints
# 4. DROP WHITE VALUES ROWS
# 5. Normalize
# 6. Plot into one graph with labels on which sign it belongs to
# 7. Use KNN

import glob

read_files = glob.glob("*.txt")

with open("combined.txt", "wb") as outfile:
    for f in read_files:
        print(f)
        with open(f, "rb") as infile:
            outfile.write(infile.read())

# Using the output from this file, it is now easy to make an excel spreadsheet called ASLalpha.csv
