import os
from tqdm import tqdm
import ray
import logging
import json
import random

ray.init(configure_logging=True, logging_level=logging.ERROR)

@ray.remote
def process_clip(label, clip, output_dir):
    video_path = clip['video_path']
    start_sec = clip['start_sec']
    end_sec = start_sec + 2  # Each clip lasts 2 seconds
    # video_filename = os.path.basename(video_path)
    video_filename = video_path.replace("/", "_")
    clip_filename = f"{label}_{start_sec:.2f}_{video_filename}"
    output_path = os.path.join(output_dir, clip_filename)

    if os.path.exists(output_path):
        # File already exists, skip processing
        print(f"File '{output_path}' already exists. Skipping.")
        return output_path, label

    os.system(
        f'ffmpeg -ss {start_sec} -t 2 -hide_banner -loglevel error -n -i "{video_path}" -vcodec copy -acodec copy "{output_path}"')

    return output_path, label

def create_csv(files_labels, output_csv):
    with open(output_csv, 'w') as f:
        for file, label in files_labels:
            f.write(f"{file},{label}\n")

if __name__ == "__main__":
    annotation_file = '/mnt/arc/yygx/pkgs_baselines/TimeSformer/process_human_data/stsb-roberta-large/match_thre0.6.json'  # Modify this path to your actual annotation file
    output_dir = '/mnt/arc/yygx/datasets/egoexo_v2_clips'
    os.makedirs(output_dir, exist_ok=True)

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    all_clips = []
    for label, data in annotations.items():
        for clip in data['matches']:
            if "None.mp4" in clip["video_path"]:
                continue
            all_clips.append((int(label), clip))

    random.shuffle(all_clips)
    num_clips = len(all_clips)
    train_split = int(0.6 * num_clips)
    val_split = int(0.2 * num_clips)

    train_clips = all_clips[:train_split]
    val_clips = all_clips[train_split:train_split + val_split]
    test_clips = all_clips[train_split + val_split:]

    train_futures = [process_clip.remote(label, clip, output_dir) for label, clip in train_clips]
    val_futures = [process_clip.remote(label, clip, output_dir) for label, clip in val_clips]
    test_futures = [process_clip.remote(label, clip, output_dir) for label, clip in test_clips]

    # Progress bars for each set
    train_results = []
    val_results = []
    test_results = []

    pbar = tqdm(total=len(train_futures), desc='Processing training clips')
    while train_futures:
        done, train_futures = ray.wait(train_futures)
        train_results.extend(ray.get(done))
        pbar.update(len(done))
    pbar.close()

    pbar = tqdm(total=len(val_futures), desc='Processing validation clips')
    while val_futures:
        done, val_futures = ray.wait(val_futures)
        val_results.extend(ray.get(done))
        pbar.update(len(done))
    pbar.close()

    pbar = tqdm(total=len(test_futures), desc='Processing test clips')
    while test_futures:
        done, test_futures = ray.wait(test_futures)
        test_results.extend(ray.get(done))
        pbar.update(len(done))
    pbar.close()

    create_csv(train_results, os.path.join(output_dir, 'train.csv'))
    create_csv(val_results, os.path.join(output_dir, 'val.csv'))
    create_csv(test_results, os.path.join(output_dir, 'test.csv'))

    ray.shutdown()
