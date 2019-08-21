import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import numpy as np
import pandas as pd
import gym,roboschool
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
"""
sample trajectory as demonstration

"""
#-----------------------------------------------
save_movie=False#!
run_dir ='path/to/model_dir'
model_itr = 1000
episodes = 2
sample_step_per_trj = 300
run_otherenv =True
otherenv = gym.make('RoboschoolAntS-v1')
#-----------------------------------------------
def load_policy(fpath, itr='last', deterministic=False):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.',model)
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action#!get action is funciton


def run_policy(env, get_action, save_dir,max_ep_len=10000, num_episodes=10, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."
    
    dir_name ='trajectory{}st_{}episode'.format(sample_step_per_trj,num_episodes)#!dirname
    dir_path = osp.join(save_dir,dir_name)
    os.makedirs(dir_path)#, exist_ok=True) #すでに存在する場合
    if save_movie:
        env =gym.wrappers.Monitor(env,dir_path+'/movies',video_callable=(lambda n: n <10))

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    observations =[]
    actions = []
    results = []
    while n < num_episodes:
        for t in range(max_ep_len):
            if render:
                env.render()
                time.sleep(1e-5)#1e-2

            a = get_action(o)
            if t < sample_step_per_trj:
                observations.append(o)
                actions.append(a)


            o, r, d, _ = env.step(a)


            ep_ret += r
            ep_len += 1

            if d or (ep_len == max_ep_len):
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
                results.append([n,ep_ret,ep_len])
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                n += 1
                break

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()

    #save csv
    df_obs = pd.DataFrame(observations)
    df_act = pd.DataFrame(actions)
    df_results = pd.DataFrame(results,columns=['Episode', 'EpRet', 'Eplen'],)
    #sample_r_mean= df_results['EpRet'].mean()
    
    

    df_obs.to_csv(osp.join(dir_path,"observations.csv"), sep=",", header=False, index=False)
    df_act.to_csv(osp.join(dir_path,"actions.csv"), sep="," ,header=False, index=False)
    df_results.to_csv(osp.join(dir_path,"each_results.csv"), sep=",",index=False)
    df_results.describe().to_csv(osp.join(dir_path,"results_describe.csv"), sep=",")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str,default =run_dir)
    parser.add_argument('--len', '-l', type=int, default=10000)
    parser.add_argument('--episodes', '-n', type=int, default=episodes)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=model_itr)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    print(args.fpath)
    env, get_action = load_policy(args.fpath, 
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic)
    if args.itr >=0:
        traj_save_dir =os.path.join( args.fpath , 'simple_save{}'.format(args.itr))
    else:
         traj_save_dir = os.path.join(args.fpath , 'simple_save')
    if run_otherenv :
        env = otherenv
    run_policy(env, get_action,traj_save_dir, args.len, args.episodes, not(args.norender))