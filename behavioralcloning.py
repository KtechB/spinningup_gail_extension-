import tensorflow as tf
import spinup.algos.ppo.core as core

import numpy as np
import spinup

import gym,roboschool
import time
from spinup.algos.ppo.si_buffer import SIBuffer

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from discriminator_js import Discriminator
from js_divergence import JS_div_machine

class BehavioralCloning:
    def __init__(self,sess, pi,logp,expert_x_ph,expert_a_ph):
        self.pi = pi
        self.logp = logp 
        self.x_ph =expert_x_ph
        self.a_ph = expert_a_ph
        self.sess =sess

        
        
        self.loss = - tf.reduce_mean(self.logp)

        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(self.loss)

    def train(self, obs, act):
        return self.sess.run(self.train_op, feed_dict={self.x_ph: obs,
                                                                      self.a_ph: act})
    def get_loss(self, obs, act):
        return self.sess.run(self.loss, feed_dict={self.x_ph: obs,
                                                                      self.a_ph: act})
                                                                    
    def learn(self,e_obs,e_acts,max_itr =1000,batch_size =128,get_loss_frq =100):
        self.dataset = Mujoco_Dset(e_obs,e_acts)
        for itr in range(max_itr):
            ob_expert, ac_expert = self.dataset.get_next_batch(batch_size, 'train')
            self.train(ob_expert,ac_expert)
            if itr%get_loss_frq ==0:
                ob_expert, ac_expert =self.dataset.get_next_batch(batch_size, 'val')
                loss = self.get_loss(ob_expert,ac_expert)
                print('{}itr_loss='.format(itr)+str(loss))# ,end =',')
    
    def learn_nobatch(self,e_obs,e_acts,max_itr =500,get_loss_frq =10):
        for itr in range(max_itr):
            ob_expert, ac_expert = e_obs,e_acts
            self.train(ob_expert,ac_expert)
            if itr%get_loss_frq ==0:
                loss = self.get_loss(ob_expert, ac_expert)
                print('{}loss='.format(itr)+loss ,end =',')


class Dset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels

class Mujoco_Dset(object):
    '''
    this is class copy from baseline of Open AI
    '''
    def __init__(self, e_obs,e_acts, train_fraction=0.7, traj_limitation=-1, randomize=True):
        '''
        traj_data = np.load(expert_path)
        if traj_limitation < 0:
            traj_limitation = len(traj_data['obs'])
        '''
        self.obs = e_obs
        self.acs = e_acts
        if traj_limitation < 0:
            trj_limitation = e_obs.shape[0]

        # obs, acs: shape (N, L, ) + S where N = # episodes, L = episode length
        # and S is the environment observation/action space.
        # Flatten to (N * L, prod(S))

        assert len(self.obs) == len(self.acs)
        self.num_traj =traj_limitation
        self.num_transition = len(self.obs)
        self.randomize = randomize
        self.dset = Dset(self.obs, self.acs, self.randomize)
        # for behavior cloning
        self.train_set = Dset(self.obs[:int(self.num_transition*train_fraction), :],
                              self.acs[:int(self.num_transition*train_fraction), :],
                              self.randomize)
        self.val_set = Dset(self.obs[int(self.num_transition*train_fraction):, :],
                            self.acs[int(self.num_transition*train_fraction):, :],
                            self.randomize)



    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError

def main(env_fn,traj_dir, actor_critic=core.mlp_actor_critic,bc_itr=1000 ,ac_kwargs=dict(),d_hidden_size =64,seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=4000,
        target_kl=0.01, save_freq=100,
        r_env_ratio=0,  reward_type = 'negative',trj_num =30, buf_size= None, 
        si_update_ratio=0.02,js_threshold_ratio = 0.5,js_smooth = 5):
    """
    test behavior cloning
    """

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    D=Discriminator(env,hidden_size = d_hidden_size)#!add Discriminator object
    D_js_m =JS_div_machine(env,hidden_size = d_hidden_size)
    
    
    e_obs = np.loadtxt(traj_dir + '/observations.csv',delimiter=',')
    e_act = np.loadtxt(traj_dir + '/actions.csv',delimiter= ',')#Demo treajectory

    Sibuffer =SIBuffer(obs_dim, act_dim, e_obs,e_act,trj_num= trj_num, max_size =buf_size,js_smooth_num= js_smooth)#!sibuf

    assert e_obs.shape[1:] == obs_dim 
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    
    sess = tf.Session()

    BC = BehavioralCloning(sess,pi,logp,x_ph,a_ph)

    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    BC.learn(Sibuffer.expert_obs,Sibuffer.expert_act,max_itr= bc_itr)
    # Sync params across processes
    start_time = time.time()
    o, r, d, ep_ret_task,ep_ret_gail, ep_len = env.reset(), 0, False, 0,0 , 0
    # Setup model saving
    
    for epoch in range(1000000):
        a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

        

        o, r, d, _ = env.step(a[0])
        env.render()
        time.sleep(1e-3)

            
            

        ep_ret_task += r
        ep_len += 1

        terminal = d or (ep_len == max_ep_len) 
        if terminal:
            print('EpRet{},EpLen{}'.format(ep_ret_task,ep_len))
            o, r, d, ep_ret_task,ep_ret_sum,ep_ret_gail, ep_len = env.reset(), 0, False, 0, 0, 0, 0
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='RoboschoolInvertedPendulumSwingupSparse2-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--d_hid', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--exp_name', type=str, default='bc')
    parser.add_argument('--traj_dir', type=str, default='/home/ryota/spinningup_data/run_data/RoboschoolInvertedPendulumSwingup-v1/RoboschoolInvertedPendulumSwingup_ppo_4000_200/RoboschoolInvertedPendulumSwingup_ppo_4000_200_s5/simple_save300/trajectory_sample200st_1episode')
    parser.add_argument('--r_env', type=float, default=0)
    parser.add_argument('--trj_num', type=int, default=30)
    parser.add_argument('--buf_size', default=None)
    parser.add_argument('--si_update_ratio', type=float, default=0.02)
    parser.add_argument('--reward_type',choices =['negative','positive','airl'] ,default='negative')
    parser.add_argument('--js_threshold_ratio', type=float, default=0.5)
    
    args = parser.parse_args()

    #mpi_fork(args.cpu)  # run parallel code with mpi

##make env dir to arrange files to each env dir

    main(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,d_hidden_size= args.d_hid, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        traj_dir =args.traj_dir, r_env_ratio = args.r_env,reward_type = args.reward_type,
        trj_num =args.trj_num, buf_size= args.buf_size,
        si_update_ratio=args.si_update_ratio,js_threshold_ratio=args.js_threshold_ratio)