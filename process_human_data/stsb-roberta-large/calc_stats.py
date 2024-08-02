import json
import re
from collections import defaultdict

# List of JSON files
json_files = [
    "match_thre0.5.json", "match_thre0.6.json", "match_thre0.7.json",
    "match_thre0.8.json", "match_thre0.9.json", "match_thre0.55.json",
    "match_thre0.65.json", "match_thre0.75.json", "match_thre0.85.json",
    "match_thre0.95.json"
]


# Function to extract the threshold value from the filename
def extract_threshold(filename):
    match = re.search(r"match_thre(\d\.\d+).json", filename)
    return float(match.group(1)) if match else None


# Sort JSON files by threshold value
json_files_sorted = sorted(json_files, key=extract_threshold)


# Function to analyze the number of unique videos for each label in a JSON file
def analyze_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    label_video_count = defaultdict(set)

    for label, details in data.items():
        for match in details['matches']:
            video_path = match['video_path']
            label_video_count[label].add(video_path)

    return {label: len(video_paths) for label, video_paths in label_video_count.items()}


# Analyzing each JSON file and storing the results
results = {}
for json_file in json_files_sorted:
    results[json_file] = analyze_json_file(json_file)

# Outputting the results in a tabular format
for json_file, label_video_stats in results.items():
    print(f"Stats for {json_file}:\n")
    print("Label\tNumber of Unique Videos")
    for label, count in sorted(label_video_stats.items(), key=lambda item: int(item[0])):
        print(f"{label}\t{count}")
    print("\n" + "-" * 40 + "\n")
