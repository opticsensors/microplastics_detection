import glob
import json
import pandas as pd

# Map single-letter color codes to full color names
color_map = {
    'K': 'Black',
    'A': 'Grey',
    'W': 'White',
    'R': 'Red',
    'O': 'Orange',
    'Y': 'Yellow',
    'G': 'Green',
    'B': 'Blue',
    'V': 'Purple',
    'P': 'Pink',
    'N': 'Brown',
    'M': 'Multicolour',
    'C': 'Colourless'
}

data_rows = []

# Loop through all .json files in the 'results' folder
for file_path in glob.glob('results/*.json'):
    with open(file_path, 'r', encoding='utf-8') as f:
        record = json.load(f)
        
        # Convert the single-letter colour to the full name
        if 'colour' in record and record['colour'] in color_map:
            record['colour'] = color_map[record['colour']]
        
        data_rows.append(record)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_rows)

# Replace decimal dots with commas for relevant numeric columns
# (Here assuming 'axis_x' and 'axis_y' are the ones that need commas)
df['axis_x'] = df['axis_x'].apply(lambda x: str(x).replace('.', ','))
df['axis_y'] = df['axis_y'].apply(lambda x: str(x).replace('.', ','))

# Export to a pipe-delimited file; 
# Excel often allows you to import with custom delimiters.
df.to_csv('./results/labels.txt', sep='\t', index=False)