#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import torch
from tqdm import tqdm
import numpy as np
import time
import os, json
from pathlib import Path

import sys 
sys.path.append("/root/autodl-tmp/VidMuse-main")

from demos.VidMuse_app import load_model, _do_predictions_for_get_video_emb


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    
    utt2mp4 = {}
    with open('{}/mp4.scp'.format(args.dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2mp4[l[0]] = l[1]

    load_model(args.model_path)
    start_time = time.time()    
    utt2video_emb = {}

    
    for utt in tqdm(utt2mp4.keys()):
        video_path = str(utt2mp4[utt])

        video_embs = _do_predictions_for_get_video_emb(
            [str(video_path)], duration=30
        )
        # print(utt, video_embs.shape)
        # video_embs = video_embs.squeeze(0).numpy().astype(np.float32) 
        if video_embs.is_cuda:
            video_embs = video_embs.cpu()
        video_embs = video_embs.numpy().astype(np.float32) 
        # print("\t",utt, video_embs.shape)
        
        dir_index = int(len(utt2video_emb.keys()) / 100000)
        save_to = '{}/utt2video_emb/{}/{}.pt'.format(args.dir, dir_index, utt)

        if not Path(save_to).parent.exists():
            Path(save_to).parent.mkdir(parents=True)
        if Path(save_to).exists():
            pass   
        else:
            torch.save(video_embs, save_to)
        utt2video_emb[utt] = save_to
        
    # torch.save(utt2video_emb, '{}/utt2video_emb.pt'.format(args.dir))
    with open('{}/utt2video_emb.json'.format(args.dir), 'w') as json_file:
        json.dump(utt2video_emb, json_file)

    logging.info('spend time {}'.format(time.time() - start_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        type=str)
    parser.add_argument('--model_path',
                        type=str, default="/root/autodl-tmp/VidMuse-main/model")
    
    args = parser.parse_args()

    main(args)

