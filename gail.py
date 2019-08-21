import numpy as np
import spinup

import tensorflow as tf
import gym,roboschool
import time
import spinup.algos.ppo.core as core
from spinup.algos.ppo.si_buffer import SIBuffer

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from behavioralcloning import BehavioralCloning
from discriminator_js import Discriminator
from js_divergence import JS_div_machine
''' this is made by ryota
copy from ppo.py  and modified it
'''
def sample_batch(data, batch_size=32):
        N = data.shape[0]
        batch_idxs = np.random.randint(0, N, batch_size)  # trajectories are negatives
        return data[batch_idxs]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.r_gail_buf = np.zeros(size, dtype=np.float32)
        
        self.slicelist=[]
        
        

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        #self.rew_buf[self.ptr] = rew#
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def store_rew(self,rew):
        assert self.ptr >=1 ,'this function must be used after store'
        self.rew_buf[self.ptr-1] = rew#

    def finish_path(self):
        """
        save path_slice and fix start_index
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        self.slicelist.append(path_slice)
        self.path_start_idx = self.ptr


    def culculate_adv_buf(self):
        for path_slice in self.slicelist:
            rews = np.insert(self.rew_buf[path_slice],0,0)#insert first zero reward 
            vals = np.append(self.val_buf[path_slice], rews[-1])#insert last value
    
            # the next two lines implement GAE-Lambda advantage calculation
            deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
            self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0# reset index
        self.slicelist = []
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]
    
    def get_trajectory(self):#!get state-action pair
        return [self.obs_buf, self.act_buf]
    
    def replace_rew_buf(self,new_rew_buf):#not used
        '''
        new_rew_buf need to be np.(dtype=np.float32)
        '''
        self.rew_buf = new_rew_buf

    def path_slice(self):
        path_slice = slice(self.path_start_idx, self.ptr)
        return path_slice

"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def gail(env_fn,traj_dir, actor_critic=core.mlp_actor_critic_add, ac_kwargs=dict(),d_hidden_size =64,d_batch_size = 64,seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=40, train_v_iters=40, lam=0.97, max_ep_len=4000,beta =1e-4,
        target_kl=0.01, logger_kwargs=dict(), save_freq=100,
        r_env_ratio=0,gail_ratio =1, d_itr =20, reward_type = 'negative',
        pretrain_bc_itr =0):
    """

    additional args
    d_hidden_size : hidden layer size of Discriminator
    d_batch_size : Discriminator's batch size

    r_env_ratio,gail_ratio : the weight of rewards from envirionment and gail .Total reward = gail_ratio *rew_gail+r_env_ratio* rew_from_environment
    
    d_itr : The number of iteration of update discriminater 
    reward_type : GAIL reward has three type ['negative','positive', 'AIRL']
    trj_num :the number of trajectory for 
    pretrain_bc_itr: the number of iteration of pretraining by behavior cloeing
    
    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    D=Discriminator(env,hidden_size = d_hidden_size,reward_type =reward_type)
    
    
    e_obs = np.loadtxt(traj_dir + '/observations.csv',delimiter=',')
    e_act = np.loadtxt(traj_dir + '/actions.csv',delimiter= ',')#Demo treajectory

    Sibuffer =SIBuffer(obs_dim, act_dim, e_obs,e_act,trj_num= 0, max_size =None)#!sibuf

    assert e_obs.shape[1:] == obs_dim 
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    adv_ph, ret_ph, logp_old_ph = core.placeholders(None, None, None)

    # Main outputs from computation graph
    pi, logp, logp_pi,pi_std, entropy, v = actor_critic(x_ph, a_ph, **ac_kwargs)

    # Need all placeholders in *this* order later (to zip with data from buffer)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, logp_old_ph]

    # Every step, get: action, value, and logprob
    get_action_ops = [pi, v, logp_pi]

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)
    #buf_gail = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)#add buffer with TRgail rewards

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))- beta*entropy
    v_loss = tf.reduce_mean((ret_ph - v)**2)#ret_phには累積報酬のバッファが入る
    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    # Optimizers
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

    sess = tf.Session()


    BC = BehavioralCloning(sess,pi,logp,x_ph,a_ph)
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())


    # Sync params across processes

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})

    def update():
        inputs = {k:v for k,v in zip(all_phs, buf.get())}#all_phsは各バッファーに対応するプレースホルダー辞書
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        # Training#ここも変える必要あり? おそらく変えなくて良い
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:#更新時のklが想定の1.5倍大きいとログをだしてtrainループを着る
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
        logger.store(StopIter=i)
        for _ in range(train_v_iters):#ｖの更新
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update（新しいロスの計算）
        pi_l_new, v_l_new, kl, cf = sess.run([pi_loss, v_loss, approx_kl, clipfrac], feed_dict=inputs)
        
        std, std_ent = sess.run([pi_std,entropy],feed_dict = inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=std_ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old),#更新での改善量
                     DeltaLossV=(v_l_new - v_l_old),
                     Std = std)

    start_time = time.time()
    o, r, d, ep_ret_task,ep_ret_gail, ep_len = env.reset(), 0, False, 0,0 , 0


    if pretrain_bc_itr>0:
        BC.learn(Sibuffer.expert_obs,Sibuffer.expert_act ,max_itr =pretrain_bc_itr)

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={x_ph: o.reshape(1,-1)})

            # save and log
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a[0])
            buf.store_rew(r)
            '''
            if t <150:
                env.render()
                time.sleep(0.03)
            '''

            ep_ret_task += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if d:# if trajectory didn't reach terminal state, bootstrap value target
                    last_val = r 
                else:
                    last_val = sess.run(v, feed_dict={x_ph: o.reshape(1,-1)})#v_last=...だったけどこれで良さげ
            
                buf.store_rew(last_val)#if its terminal ,nothing change and if its maxitr last_val is use
                buf.finish_path()
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret_task, EpLen=ep_len)#,EpRet_Sum =ep_ret_sum,EpRet_Gail =ep_ret_gail)
        
                o, r, d, ep_ret_task,ep_ret_sum,ep_ret_gail, ep_len = env.reset(), 0, False, 0, 0, 0, 0

        # Save model
        
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)


        agent_obs , agent_act = buf.obs_buf, buf.act_buf

        d_batch_size = d_batch_size#or len(agent_obs)//d_itr #update discreminator
        for _t in range(d_itr):
            e_obs_batch ,e_act_batch =Sibuffer.get_random_batch(d_batch_size)
            a_obs_batch  =sample_batch(agent_obs,batch_size = d_batch_size)
            a_act_batch= sample_batch(agent_act,batch_size = d_batch_size)
            D.train(sess, e_obs_batch,e_act_batch , a_obs_batch,a_act_batch )
        js_d = D.get_js_div(sess,Sibuffer.main_obs_buf,Sibuffer.main_act_buf,agent_obs,agent_act)
        #---------------get_gail_reward------------------------------
        rew_gail=D.get_reward(sess,agent_obs, agent_act).ravel()

        buf.rew_buf = gail_ratio *rew_gail+r_env_ratio*buf.rew_buf
        for path_slice in buf.slicelist[:-1]:
            ep_ret_gail = rew_gail[path_slice].sum()
            ep_ret_sum = buf.rew_buf[path_slice].sum()
            logger.store(EpRet_Sum=ep_ret_sum,EpRet_Gail=ep_ret_gail)


        buf.culculate_adv_buf()
        
        # -------------Perform PPO update!--------------------

        update()
        
        logger.store(JS=js_d)


        # Log info about epoch
        #if epoch%10 == 0:#logger print each 10 epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRet_Sum', average_only=True)
        logger.log_tabular('EpRet_Gail', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.log_tabular('Std', average_only=True)
        logger.log_tabular('JS', average_only=True)
        #logger.log_tabular('JS_Ratio', average_only=True)    
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='RoboschoolAntSparse-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--d_hid', type=int, default=64)
    parser.add_argument('--d_batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--logvar', type=float, default=-0.4)#the smaller logvar ,default std get smaller
    parser.add_argument('--beta', type=float, default=1e-5)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--exp_name', type=str, default='GailrllabAntSparse100_4epdemo')
    parser.add_argument('--traj_dir', type=str, default='/home/ryota/spinningup_data/run_data/RoboschoolAnt-v1/RoboschoolAnt-v1_ppo_ent0.001/RoboschoolAnt-v1_ppo_ent0.001_s5/simple_save500/trajectory100st_4episode')
    parser.add_argument('--r_env', type=float, default=1)
    parser.add_argument('--gail_ratio', type=float, default=1)
    parser.add_argument('--g_itr', type=int, default=40)
    parser.add_argument('--d_itr', type=int, default=10)
    parser.add_argument('--reward_type',choices =['negative','positive','airl'] ,default='positive')
    parser.add_argument('--bc_itr', type=int, default=500)
    
    
    args = parser.parse_args()

    #mpi_fork(args.cpu)  # run parallel code with mpi

##make env dir to arrange files to each env dir
    from spinup.user_config import DEFAULT_DATA_DIR 
    import os

    datadir= os.path.join(DEFAULT_DATA_DIR ,args.env)
    os.makedirs(datadir, exist_ok=True)
##

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir= datadir)

    gail(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic_add,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l,policy_logvar=args.logvar), gamma=args.gamma,d_hidden_size= args.d_hid, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,train_pi_iters =args.g_itr,train_v_iters = args.g_itr,d_batch_size = args.d_batch_size,
        beta = args.beta,logger_kwargs=logger_kwargs , traj_dir =args.traj_dir, r_env_ratio = args.r_env,gail_ratio= args.gail_ratio,d_itr = args.d_itr,reward_type = args.reward_type,
        pretrain_bc_itr =args.bc_itr)