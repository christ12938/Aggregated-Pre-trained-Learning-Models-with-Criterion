import pandas as pd
import json
import argparse
from utils import create_folder
import ast


parser = argparse.ArgumentParser(description='Process input file.')

# Add an argument for the input file
parser.add_argument('-i', '--input', required=True, help='Input file name')
parser.add_argument('-o', '--output', required=True, help='Output file name')
parser.add_argument('-m', '--mode', required=True, help='Mode')


# Parse the command-line arguments
args = parser.parse_args()

# Retrieve the input file name from the parsed arguments
input_file = args.input
output_file = args.output
mode = args.mode

create_folder(path=output_file)

# Perform Conversion
if mode == "vocab" or mode == "doc":
    df = pd.read_pickle(input_file) 
    json_dict = {}
    for entry in df.to_dict("records"):
        if mode == "vocab":
            json_dict[entry["vocab"]] = {"id": entry["id"], "count": int(entry["count"]), 'idf': float(entry['standardized_idf'])}
        elif mode == "doc":
            json_dict[entry["id"]] = int(entry["length"])

    # Write File
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)

elif mode == "jsonl":
    data = []
    with open(input_file, "r", encoding='utf-8') as f:  # Assuming a .jsonl file extension for JSON lines
        for line in f:
            data.append(ast.literal_eval(line.strip()))
    pd.DataFrame(data).to_pickle(output_file)
