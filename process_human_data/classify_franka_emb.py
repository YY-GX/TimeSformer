import copy
import json
import pickle
from pathlib import Path
import os
import random
from tqdm import tqdm
import torch
from collections import defaultdict
import pdb

from sentence_transformers import util as st_utils
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import transformers


translation_lm_id = "stsb-roberta-large"
# translation_lm_id = "all-mpnet-base-v2"
# translation_lm_id = "clip-ViT-L-14"
translation_lm = SentenceTransformer(translation_lm_id)


def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True)
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0]
    most_similar_idx, matching_score = int(torch.argmax(cos_scores)), float(torch.max(cos_scores))
    return most_similar_idx, matching_score


def load_pkl(fn):
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, fn):
    with open(fn, 'wb') as f:
        pickle.dump(data, f)

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

def get_franka_descriptions():
    res = [
        "Turn the oven knob that activates the bottom burner.",
        "Turn the oven knob that activates the top burner.",
        "Turn on the light switch.",
        "Open the slide cabinet.",
        "Open the left hinge cabinet.",
        "Open the microwave door.",
        "Move the kettle to the top left burner.",
    ]
    return res


def sim_take():
    # take_anno = load_json('/mnt/opr/ce/datasets/egoexo/v1/takes.json')
    take_anno = load_json('/mnt/mir/datasets/egoexo/v2/takes.json')
    '''
    take_uid --> {
        task
        root_url
    }
    '''
    res = {}
    for el in take_anno:
        res[el['take_uid']] = {
            "task": el['parent_task_name'],
            "root_dir": el['root_dir']
        }
    save_json(res, 'egoexo_v2/take_sim.json')


def get_cook_anno():
    # take_anno = load_json('take_sim.json')
    take_anno = load_json('egoexo_v2/take_sim.json')

    # data = load_json('/mnt/opr/ce/datasets/egoexo/annotations/atomic_descriptions_latest.json')
    data = load_json('/mnt/mir/datasets/egoexo/v2/annotations/atomic_descriptions_train.json')['annotations']
    data.update(load_json('/mnt/mir/datasets/egoexo/v2/annotations/atomic_descriptions_val.json')['annotations'])

    cook_uids = set([k for k, v in take_anno.items() if 'ook' in v['task']])

    print(len(list(cook_uids)))  # 636   678
    print(list(cook_uids)[:10])

    res = {k: v for k, v in data.items() if k in cook_uids}
    save_json(res, 'egoexo_v2/anno_cook.json')
    

def match():
    franka_descriptions = get_franka_descriptions()
    anno_all = load_json('egoexo_v2/anno_cook.json')
    take_info = load_json('egoexo_v2/take_sim.json')
    output_path = f'egoexo_v2/{translation_lm_id}/match.json'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    base_video_path = '/mnt/mir/datasets/egoexo/v2'

    # get list of annos
    anno_list = []
    for take_uid, v in anno_all.items():
        for text_info in v[0]['descriptions']:
            anno = {
                "root_dir": take_info[take_uid]['root_dir'],
                "video_path": "",
                'start_sec': text_info['timestamp'],
                'end_sec': -2,
                'narr': text_info['text']
            }
            if text_info['best_exo'] is not None:
                anno['video_path'] = os.path.join(base_video_path, take_info[take_uid]['root_dir'], 'frame_aligned_videos/downscaled/448', f"{text_info['best_exo']['cam_id']}.mp4")
                anno_list.append(anno)

    corpus_embs = translation_lm.encode(franka_descriptions, batch_size=32, convert_to_tensor=True)
    # for each human anno
    for anno in tqdm(anno_list):
        matched_narr_id, score = find_most_similar(anno['narr'], corpus_embs)
        anno['matched_narr'] = franka_descriptions[matched_narr_id]
        anno['matched_narr_id'] = matched_narr_id
        anno['confidence'] = score
    
    # group by robot narrh
    res = {}
    for i, narr in enumerate(franka_descriptions):
        res[i] = {
            "narr": narr,
            "id": i,
            "matches": []
        }
    for anno in anno_list:
        res[anno['matched_narr_id']]['matches'].append({
            "root_dir": anno['root_dir'],
            'video_path': anno['video_path'],
            'start_sec': anno['start_sec'],
            'end_sec': anno['end_sec'],
            'narr': anno['narr'],
            'confidence': anno['confidence'],
        })
    save_json(res, output_path)


def filter(thre):
    output_path = f'egoexo_v2/{translation_lm_id}/match_thre{thre}.json'
    data = load_json(f'egoexo_v2/{translation_lm_id}/match.json')
    output = copy.deepcopy(data)
    for robot_narr_id, info in data.items():
        matches_new = [el for el in info['matches'] if el['confidence'] >= thre]
        output[robot_narr_id]['matches'] = matches_new
    save_json(output, output_path)
    stat_from_path(output_path)

def filter_launch():
    thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # thres = [0.7, 0.75, 0.8, 0.85, 0.9]
    for thre in thres:
        print('thre: ', thre)
        filter(thre)
        print()

def stat_from_path(p):
    data = load_json(p)
    for k, v in data.items():
        print(k, len(v['matches']))


if __name__ == '__main__':
    # sim_take()
    # get_cook_anno()
    # match()
    # filter()
    filter_launch()


# list of human anno
'''
[
    {
        "root_dir": xxx, # egoexo-only, this is a list
        "video_path": xxx,   # "" if no best_view in egoexo
        "start_sec": xxx,
        "end_sec": xxx,  # -2 if no end-sec
        "narr": xxx,
        "matched_narr": xxx,
        "matched_narr_id": xxx,
        "confidence": xxx,
    },
    {},
    {},
]
'''


'''
{
    id: {
        "narr": xxx,
        "id": xxx,
        "matches": [
            {
                "root_dir": xxx, # egoexo-only, this is a list
                "video_path": xxx,   # "" if no best_view in egoexo
                "start_sec": xxx,
                "end_sec": xxx,  # -2 if no end-sec
                "narr": xxx,
                "confidence": xxx,
            },
            {},  # for one narration in human dataset
            {}
        ]
    },
    id: {},   # for one narration in robot dataset ; id is robot narr id
    id: {}
}
'''



'''
thre:  0.5
0 1024
1 1146
2 72
3 191
4 669
5 225
6 324

thre:  0.55
0 468
1 652
2 35
3 89
4 408
5 119
6 115

thre:  0.6
0 44
1 261
2 8
3 31
4 298
5 27
6 25

thre:  0.65
0 1
1 31
2 3
3 16
4 194
5 6
6 3

thre:  0.7
0 0
1 0
2 0
3 10
4 99
5 4
6 1
'''