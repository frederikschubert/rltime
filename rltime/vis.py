""" Entry point for visualizing a trained policy. """

import argparse
import json
import os
import numpy as np
import time
import datetime

from rltime.general.config import load_config
from rltime.general.utils import deep_dictionary_update
from rltime.general.type_registry import get_registered_type
from rltime.env_wrappers.common import make_env_creator, EpisodeRecorder
from rltime.env_wrappers.vec_env.sub_proc import make_sub_proc_vec_env
from rltime.general.loggers import DirectoryLogger


from rltime.eval import create_policy_from_config


def vis_policy(path, num_envs, episode_count, record=False, record_fps=60,
                render=False, render_fps=None, eps=0.001, conf_update=None):
    # TODO(frederik): Implement wrapper to return colored and preprocessed observation
    # TODO(frederik): Perform steps in the environment collecting the obs and colored obs
    # TODO(frederik): Use https://github.com/utkuozbulak/pytorch-cnn-visualizations to visualize the policy
    # TODO(frederik): Persist saliency maps to output directory (see https://github.com/greydanus/visualize_atari)
    pass

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path', type=str,
        help="The path to the training directory result to evaluate")
    parser.add_argument(
        '--num-envs', type=int, default=1,
        help="Amount of ENVs to run in parallel")
    parser.add_argument(
        '--episodes', type=int, default=5,
        help="Amount of episodes to run")
    parser.add_argument(
        '--record', action='store_true',
        help="Whether to record episode to MP4 (To a sub-directory in the "
             "result path). Warning: If used with --num-envs>1 the last "
             "videos will be truncated")
    parser.add_argument(
        '--record-fps', type=int, default=60,
        help="FPS to record at if --record (Typically 60FPS for atari)")
    parser.add_argument(
        '--render', action='store_true',
        help="Whether to render the episodes in real-time")
    parser.add_argument(
        '--render-fps', type=int, default=0,
        help="FPS to sync to if using --render (Set to 0 for full speed), "
        "note this is after ENV frame-skipping so if you want 60FPS with "
        "frame-skip of 4 use 15 here")
    parser.add_argument(
        '--eps', type=float, default=0.001,
        help="Epsilon value to use for random action selection during "
             "evaluation")
    parser.add_argument(
        '--conf-update', type=str,
        help="Optional JSON dictionary string to deep-update the config with")
    return parser.parse_args()


def main():
    args = parse_args()
    conf_update = None if not args.conf_update \
        else json.loads(args.conf_update)

    vis_policy(
        args.path, num_envs=args.num_envs, episode_count=args.episodes,
        record=args.record, record_fps=args.record_fps,
        render=args.render, render_fps=args.render_fps, eps=args.eps, conf_update=conf_update)

if __name__ == '__main__':
    main()