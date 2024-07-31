import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import ray
import logging

columns = [
    'id', 'action_id', 'action_name', 'player_id', 'player_name',
    'team_id', 'team_name', 'opponent_id', 'opponent_name',
    'opponent_team_id', 'opponent_team_name', 'teammate_id',
    'teammate_name', 'half', 'second', 'pos_x', 'pos_y',
    'possession_id', 'possession_name', 'possession_team_id',
    'possession_team_name', 'possession_number', 'possession_start_clear',
    'possession_end_clear', 'playtype', 'hand', 'shot_type',
    'drive', 'dribble_move', 'contesting', 'ts'
]

ray.init(configure_logging=True,logging_level=logging.ERROR)

@ray.remote
def process_video(video, video_path, log_path, save_path, matching_list, ignore, season):
    if video == '.DS_Store':
        return

    game_id = int(video.split('_')[0])
    curr_video = os.path.join(video_path, video)
    #tqdm.write(f'Working on: {video}')

    matching_row = matching_list.loc[matching_list['id'].values == game_id]
    if matching_row.empty:
        #print("Missing log!!! {video}")
        tqdm.write(f"Missing log!!! {video}")
        with open(f'{season}_log_missing.txt', 'a+') as f:
            f.write(f'{curr_video}\n')
        return
    else:
        matching_row = matching_row.iloc[0]

    game_path = os.path.join(log_path, matching_row['season'], matching_row['league'], matching_row['log_name'])

    if os.path.exists(game_path):
        log_df = pd.read_csv(game_path, skiprows=1, delimiter=';', header=None, names=columns)
    else:
        with open(f'{season}_log_missing.txt', 'a+') as f:
            f.write(f'{curr_video}\n')
        tqdm.write(f'log missing: {video}')
        return

    extend_time = 3.5

    period = int(video[-5])
    for index, row in log_df.iterrows():
        if int(row['half']) != period:
            continue

        if row['action_name'] not in ignore and not pd.isna(row['second']) and not pd.isna(row['player_name']):
            if row['action_name'] == 'Assisting':
                if pd.isna(row['teammate_id']):
                    continue
                player_id = int(row['teammate_id'])
                player_name = row['teammate_name']
                duration = 10
            else:
                if '1' in row['action_name']:
                    extend_time = 4.5
                elif row['action_name'] == 'Turnover':
                    extend_time = 2.5
                player_id = int(row['player_id'])
                player_name = row['player_name']
                duration = 10
            
            start_time = float(row['second']) + extend_time

            output_folder = os.path.join(save_path, matching_row['season'], matching_row['league'], str(player_id))
            #print(output_folder)
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, f"{player_id}_{player_name}_{row['action_name']}_{row['id']}.mp4")

            if not os.path.exists(output_path):
                os.system(
                    f'ffmpeg -ss {start_time - duration} -t {duration} -hide_banner -loglevel error -n -i "{curr_video}" -vcodec copy -acodec copy "{output_path}"')
                # os.system(
                #     f'ffmpeg -ss {start_time - duration} -t {duration} -hide_banner -loglevel error -n -i "{curr_video}" -filter:v scale="trunc(oh*a/2)*2:256" -c:a copy "{output_path}"')
            
            if not os.path.exists(output_path):
                with open(f'{season}_failed_videos.txt', 'a+') as f:
                    f.write(f'{curr_video}\n')
                return False

    return True


if __name__ == "__main__":
    season = '22-23'
    video_path = f'/Volumes/Seagate_18TB_23X_HUDL_data_21-23/raw_videos/{season}/400'
    matching_list = pd.read_csv(f'{season}_game_id_matching.csv')
    log_path = '/Users/yulupan/Desktop/Active_Project/basketball_skill_analysis/game_logs'
    save_path = f'/Volumes/Seagate_18TB_23X_HUDL_data_ALL_clips_13_24/{season}_all_clip'
    os.makedirs(save_path, exist_ok=True)

    ignore = ['Start of the offensive possession', 'Shooting guard', 'Guard', 'Center', 'Power forward', 'Forward',
              'Timeout', 'Halftime', '2nd quarter', 'Starting lineup', '3rd quarter', '1st quarter', '4th quarter',
              'Match end', 'Game stop', 'Ball in play']
    
    ignore.append('Error leading to goal')
    ignore.append('Accurate pass')

    videos = [video for video in os.listdir(video_path) if video != '.DS_Store']
    
    pbar = tqdm(total=len(videos))
    matching_list = ray.put(matching_list)
    video_path = ray.put(video_path)
    ignore = ray.put(ignore)
    save_path = ray.put(save_path)
    log_path = ray.put(log_path)
    seaon = ray.put(season)

    futures = [process_video.remote(video, video_path, log_path, save_path, matching_list, ignore, season) for video in videos]

    num_failed = 0
    while len(futures):
        dones, futures = ray.wait(futures)
        for done in dones:
            if ray.get(done) == False:
                num_failed+=1
            if num_failed % 100 == 0 and num_failed > 0:
                print(f'# failed video: {num_failed}')
        pbar.update(len(dones))

    print(f'Total # failed video: {num_failed}')

    ray.shutdown()
