#!/usr/bin/env python3

import logging
import os
from os import path as osp
import sys
import time
from multiprocessing import Process, Queue

import cloudpickle
import easy_tf_log
from drlhp.deprecated.a2c import logger, learn
from drlhp.deprecated.a2c import CnnPolicy, MlpPolicy
from drlhp.deprecated.params import parse_args, PREFS_VAL_FRACTION
from drlhp.pref_db import PrefDB, PrefBuffer
from drlhp.pref_interface import PrefInterface
from drlhp.reward_predictor import RewardPredictorEnsemble
from drlhp.reward_predictor_core_network import net_cnn, net_moving_dot_features
from drlhp.utils import VideoRenderer, get_port_range, make_envs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def main():
    general_params, a2c_params, \
        pref_interface_params, rew_pred_training_params = parse_args()

    if general_params['debug']:
        logging.getLogger().setLevel(logging.DEBUG)

    run(general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_training_params)


def run(general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_training_params):

    # SANITY CHECKS
    assert ('env' in a2c_params) != (
                'env_id' in a2c_params), "At least and only one of env or env_id should be passed in"


    # CREATE COORDINATION QUEUES
    seg_pipe = Queue(maxsize=1)  # Passed into the Runner, where it's used by update_segment_buffer
    pref_pipe = Queue(maxsize=1) # Feeds into PrefBuffer, which feeds into PrefDB
    start_policy_training_flag = Queue(maxsize=1)


    # CREATE VIDEO_RENDERER and EPISODE_VID_QUEUE
    # This works by creating a queue and then having that be the input process for the VideoRenderer, and then
    # passing back both the renderer and the queue that feeds it
    if general_params['render_episodes']:
        # episode_vid_queue gets added to by Runner.update_episode_frame_buffer
        episode_vid_queue, episode_renderer = start_episode_renderer()
    else:
        episode_vid_queue = episode_renderer = None


    # CHOOSE REWARD PREDICTOR
    # Determine which reward predictor network (a function) to use, and create the make_reward_predictor() function
    reward_predictor_network = rew_pred_training_params.get('reward_predictor_network')
    if reward_predictor_network is None:
        if a2c_params['env_id'] in ['MovingDot-v0', 'MovingDotNoFrameskip-v0']:
            reward_predictor_network = net_moving_dot_features
        elif a2c_params['env_id'] in ['PongNoFrameskip-v4', 'EnduroNoFrameskip-v4']:
            reward_predictor_network = net_cnn
        else:
            raise Exception("Unsure about reward predictor network for {}".format(
                a2c_params['env_id']))

    def make_reward_predictor(name, cluster_dict):
        return RewardPredictorEnsemble(
            cluster_job_name=name,
            cluster_dict=cluster_dict,
            log_dir=general_params['log_dir'],
            batchnorm=rew_pred_training_params['batchnorm'],
            dropout=rew_pred_training_params['dropout'],
            lr=rew_pred_training_params['lr'],
            core_network=reward_predictor_network,
            obs_shape=rew_pred_training_params['obs_shape'])

    save_make_reward_predictor(general_params['log_dir'],
                               make_reward_predictor)



    # CASE: GATHER INITIAL PREFERENCES
    # My read of what's happening here is that we call start_policy_training because this triggers the Runner to start
    # running and collecting episodes that can be used in the pref interface

    if general_params['mode'] == 'gather_initial_prefs':
        env, a2c_proc = start_policy_training(
            cluster_dict=None,
            make_reward_predictor=None,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params['log_dir'],
            **pref_interface_params)

        n_train = general_params['max_prefs'] * (1 - PREFS_VAL_FRACTION)
        n_val = general_params['max_prefs'] * PREFS_VAL_FRACTION
        pref_db_train = PrefDB(maxlen=n_train)
        pref_db_val = PrefDB(maxlen=n_val)
        pref_buffer = PrefBuffer(db_train=pref_db_train, db_val=pref_db_val)
        pref_buffer.start_recv_thread(pref_pipe)
        pref_buffer.wait_until_len(general_params['n_initial_prefs'])
        pref_db_train, pref_db_val = pref_buffer.get_dbs()

        save_prefs(general_params['log_dir'], pref_db_train, pref_db_val)

        pi_proc.terminate()
        pi.stop_renderer()
        a2c_proc.terminate()
        pref_buffer.stop_recv_thread()
        env.close()


    # CASE: ASSUMES prefs_dir EXISTS AND PRETRAINS REWARD PREDICTOR
    elif general_params['mode'] == 'pretrain_reward_predictor':
        cluster_dict = create_cluster_dict(['ps', 'train'])
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=True,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params['max_prefs'],
            prefs_dir=general_params['prefs_dir'],
            load_ckpt_dir=None,
            n_initial_prefs=general_params['n_initial_prefs'],
            n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
            val_interval=rew_pred_training_params['val_interval'],
            ckpt_interval=rew_pred_training_params['ckpt_interval'],
            log_dir=general_params['log_dir'])
        rpt_proc.join()
        ps_proc.terminate()

    # TRAIN POLICY FROM NORMAL REWARDS FOR COMPARISON
    elif general_params['mode'] == 'train_policy_with_original_rewards':
        env, a2c_proc = start_policy_training(
            cluster_dict=None,
            make_reward_predictor=None,
            gen_segments=False,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        start_policy_training_flag.put(True)
        a2c_proc.join()
        env.close()

    elif general_params['mode'] == 'train_policy_with_preferences':
        cluster_dict = create_cluster_dict(['ps', 'a2c', 'train'])
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        env, a2c_proc = start_policy_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params['log_dir'],
            **pref_interface_params)
        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=False,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params['max_prefs'],
            prefs_dir=general_params['prefs_dir'],
            load_ckpt_dir=rew_pred_training_params['load_ckpt_dir'],
            n_initial_prefs=general_params['n_initial_prefs'],
            n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
            val_interval=rew_pred_training_params['val_interval'],
            ckpt_interval=rew_pred_training_params['ckpt_interval'],
            log_dir=general_params['log_dir'])
        # We wait for A2C to complete the specified number of policy training
        # steps
        a2c_proc.join()
        rpt_proc.terminate()
        pi_proc.terminate()
        pi.stop_renderer()
        ps_proc.terminate()
        env.close()
    else:
        raise Exception("Unknown mode: {}".format(general_params['mode']))

    if episode_renderer:
        episode_renderer.stop()

# SAVE OUT PREFDBs
def save_prefs(log_dir, pref_db_train, pref_db_val):
    train_path = osp.join(log_dir, 'train.pkl.gz')
    pref_db_train.save(train_path)
    print("Saved training preferences to '{}'".format(train_path))
    val_path = osp.join(log_dir, 'val.pkl.gz')
    pref_db_val.save(val_path)
    print("Saved validation preferences to '{}'".format(val_path))

# SAVE OUT FUNCTION TO MAKE REWARD PREDICTOR
def save_make_reward_predictor(log_dir, make_reward_predictor):
    save_dir = osp.join(log_dir, 'reward_predictor_checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, 'make_reward_predictor.pkl'), 'wb') as fh:
        fh.write(cloudpickle.dumps(make_reward_predictor))


# SOMETHING WITH MULTIPROCESSING?
# What is a cluster dict....
def create_cluster_dict(jobs):
    n_ports = len(jobs) + 1
    ports = get_port_range(start_port=2200,
                           n_ports=n_ports,
                           random_stagger=True)
    cluster_dict = {}
    for part, port in zip(jobs, ports):
        cluster_dict[part] = ['localhost:{}'.format(port)]
    return cluster_dict


def configure_a2c_logger(log_dir):
    a2c_dir = osp.join(log_dir, 'a2c')
    os.makedirs(a2c_dir)
    tb = logger.TensorBoardOutputFormat(a2c_dir)
    logger.Logger.CURRENT = logger.Logger(dir=a2c_dir, output_formats=[tb])

# IDK WHAT THIS IS DOING
def start_parameter_server(cluster_dict, make_reward_predictor):
    def f():
        make_reward_predictor('ps', cluster_dict)
        while True:
            time.sleep(1.0)

    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_policy_training(cluster_dict, make_reward_predictor, gen_segments,
                          start_policy_training_pipe, seg_pipe,
                          episode_vid_queue, log_dir, a2c_params):

    policy_fn = a2c_params.get('policy_network')
    if policy_fn is None:
        env_id = a2c_params['env_id']
        if env_id in ['MovingDotNoFrameskip-v0', 'MovingDot-v0']:
            policy_fn = MlpPolicy
        elif env_id in ['PongNoFrameskip-v4', 'EnduroNoFrameskip-v4']:
            policy_fn = CnnPolicy
        else:
            msg = "Unsure about policy network for {}".format(a2c_params['env_id'])
            raise Exception(msg)

    configure_a2c_logger(log_dir)

    # Done here because daemonic processes can't have children
    env = make_envs(a2c_params.get('env'),
                    a2c_params.get('env_id'),
                    a2c_params.get('n_envs'),
                    a2c_params.get('seed'))

    for k in ['env_id', 'env', 'n_envs', 'policy_network']:
        if k in a2c_params:
            del a2c_params[k]

    ckpt_dir = osp.join(log_dir, 'policy_checkpoints')
    os.makedirs(ckpt_dir)

    def f():
        if make_reward_predictor:
            reward_predictor = make_reward_predictor('a2c', cluster_dict)
        else:
            reward_predictor = None
        misc_logs_dir = osp.join(log_dir, 'a2c_misc')
        easy_tf_log.set_dir(misc_logs_dir)
        learn(
            policy=policy_fn,
            env=env,
            seg_pipe=seg_pipe,
            start_policy_training_pipe=start_policy_training_pipe,
            episode_vid_queue=episode_vid_queue,
            reward_predictor=reward_predictor,
            ckpt_save_dir=ckpt_dir,
            gen_segments=gen_segments,
            **a2c_params)

    proc = Process(target=f, daemon=True)
    proc.start()
    return env, proc


def start_pref_interface(seg_pipe, pref_pipe, max_segs, synthetic_prefs,
                         log_dir, zoom, channels):
    def f():
        # The preference interface needs to get input from stdin. stdin is
        # automatically closed at the beginning of child processes in Python,
        # so this is a bit of a hack, but it seems to be fine.
        sys.stdin = os.fdopen(0)
        pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)

    # Needs to be done in the main process because does GUI setup work
    prefs_log_dir = osp.join(log_dir, 'pref_interface')
    pi = PrefInterface(synthetic_prefs=synthetic_prefs,
                       max_segs=max_segs,
                       log_dir=prefs_log_dir,
                       channels=channels,
                       zoom=zoom)
    print("Preference interface has been created")
    proc = Process(target=f, daemon=True)
    proc.start()
    return pi, proc


def start_reward_predictor_training(cluster_dict,
                                    make_reward_predictor,
                                    just_pretrain,
                                    pref_pipe,
                                    start_policy_training_pipe,
                                    max_prefs,
                                    n_initial_prefs,
                                    n_initial_epochs,
                                    prefs_dir,
                                    load_ckpt_dir,
                                    val_interval,
                                    ckpt_interval,
                                    log_dir):
    def f():
        rew_pred = make_reward_predictor('train', cluster_dict)
        rew_pred.init_network(load_ckpt_dir)
        print("Reward predictor initialized")
        if prefs_dir is not None:
            train_path = osp.join(prefs_dir, 'train.pkl.gz')
            pref_db_train = PrefDB.load(train_path)
            print("Loaded training preferences from '{}'".format(train_path))
            n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))

            val_path = osp.join(prefs_dir, 'val.pkl.gz')
            pref_db_val = PrefDB.load(val_path)
            print("Loaded validation preferences from '{}'".format(val_path))
            n_prefs, n_segs = len(pref_db_val), len(pref_db_val.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))
        else:
            n_train = max_prefs * (1 - PREFS_VAL_FRACTION)
            n_val = max_prefs * PREFS_VAL_FRACTION
            pref_db_train = PrefDB(maxlen=n_train)
            pref_db_val = PrefDB(maxlen=n_val)

        pref_buffer = PrefBuffer(db_train=pref_db_train,
                                 db_val=pref_db_val)
        pref_buffer.start_recv_thread(pref_pipe)
        if prefs_dir is None:
            pref_buffer.wait_until_len(n_initial_prefs)

        save_prefs(log_dir, pref_db_train, pref_db_val)

        if not load_ckpt_dir:
            print("Pretraining reward predictor for {} epochs".format(
                n_initial_epochs))
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
            for i in range(n_initial_epochs):
                # Note that we deliberately don't update the preferences
                # databases during pretraining to keep the number of
                # fairly preferences small so that pretraining doesn't take too
                # long.
                print("Reward predictor training epoch {}".format(i))
                rew_pred.train(pref_db_train, pref_db_val, val_interval)
                if i and i % ckpt_interval == 0:
                    rew_pred.save()
            print("Reward predictor pretraining done")
            rew_pred.save()

        if just_pretrain:
            return

        start_policy_training_pipe.put(True)
        
        i = 0
        while True:
            time.sleep(30)
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
            save_prefs(log_dir, pref_db_train, pref_db_val)
            rew_pred.train(pref_db_train, pref_db_val, val_interval)
            if i and i % ckpt_interval == 0:
                rew_pred.save()
    print("Got inside reward predictor training")
    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_episode_renderer():
    episode_vid_queue = Queue()
    renderer = VideoRenderer(
        episode_vid_queue,
        playback_speed=2,
        zoom=2,
        mode=VideoRenderer.play_through_mode)
    return episode_vid_queue, renderer


if __name__ == '__main__':
    main()
