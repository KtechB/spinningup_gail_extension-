import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Box, Discrete
import pandas as pd

"""
for make this program 
I refferd https://github.com/uidilr/gail_ppo_tf/blob/master/network_models/discriminator.py
"""

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""

def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

#######from core for every dimention envs
def combined_shape(length, shape=None):#例えばactが1次元のときaction_space.shape=()と空集合となるためうまくやる
    if shape is None:
        return (length,)
    elif shape == ():
        return (length,1)#!(none,1)と１次元化
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))
'''
def placeholder_from_space(space):
    if isinstance(space, Box):#連続値なら、、、
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.float32, shape=(None,))#離散値なら１次元？？ #!debugのためint　→　floatに変更
    raise NotImplementedError
'''
##########

def proper_shape(arr):#!input numpyarray
    if arr.ndim==1:
        return np.reshape(arr,(arr.shape[0],1))
    return arr



class Discriminator():
    def __init__(self,env, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary", reward_type='negative'):
        
        with tf.variable_scope(scope):
            self.scope = scope
            self.observation_shape = env.observation_space.shape
            self.actions_shape = env.action_space.shape
            self.hidden_size = hidden_size
            self.build_ph()

            

            # Build grpah
            with tf.variable_scope('network') as network_scope:
                generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)#generatorはAgentの行動である確率
                expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)#generatorのグラフをコピー
            # Build accuracy
            with tf.variable_scope('loss'):
                generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))#agentの行動がEXPERTである確率(Dの判断)が0.5以下なら１、それ以上なら０（正解）としてDの正解率を計算
                expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
                # Build regression loss
                # let x = logits, z = targets.
                # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
                generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))#labelはすべて0(全て間違い)
                generator_loss = tf.reduce_mean(generator_loss)
                expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))#すべて正解
                expert_loss = tf.reduce_mean(expert_loss)
                # Build entropy loss
                logits = tf.concat([generator_logits, expert_logits], 0)
                entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
                entropy_loss = -entcoeff*entropy#エントロピー項をentcoeff倍（影響を調節）
                # Loss + Accuracy terms
                self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
                self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
                self.total_loss = generator_loss + expert_loss + entropy_loss
            
            optimizer = tf.train.AdamOptimizer()
            self.train_op = optimizer.minimize(self.total_loss)
            
            # Build Reward for policy
            #self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)#つねに正の報酬
            log_d_g= tf.reduce_mean(tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8))
            log_1_d_e= tf.reduce_mean(tf.log(tf.nn.sigmoid(expert_logits)+1e-8))
            self.js_div =  (log_d_g + log_1_d_e)/2.0+ tf.log(2.0)
            #self.js_div = tf.clip_by_value(self.js_div,1e-8,1.0)#!dontclip
            self.reward_op_negat = tf.log(tf.nn.sigmoid(generator_logits)+1e-8)
            self.reward_op_posit = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
            self.reward_op_airl = self.reward_op_negat+ self.reward_op_posit

            reward_type_dict = {'negative':self.reward_op_negat, 'positive':self.reward_op_posit,'airl':self.reward_op_airl }
            assert reward_type in reward_type_dict,'reward_type is not in type_dict'
            
            self.reward_op = reward_type_dict[reward_type]
            
        
    def train(self,session,e_obs_buf,e_act_buf,a_obs_buf, a_act_buf):
        return session.run(self.train_op, feed_dict={self.generator_obs_ph:proper_shape(a_obs_buf),
                                                                    self.generator_acs_ph:proper_shape(a_act_buf),
                                                                    self.expert_obs_ph:proper_shape(e_obs_buf),
                                                                    self.expert_acs_ph:proper_shape(e_act_buf)})
    
    def get_js_div(self,session,e_obs_buf,e_act_buf,a_obs_buf, a_act_buf):
        return session.run(self.js_div, feed_dict={self.generator_obs_ph:proper_shape(a_obs_buf),
                                                                    self.generator_acs_ph:proper_shape(a_act_buf),
                                                                    self.expert_obs_ph:proper_shape(e_obs_buf),
                                                                    self.expert_acs_ph:proper_shape(e_act_buf)})
    
    

    def get_reward_buf(self,session, a_obs_buf, a_act_buf):#get_reward
        return session.run(self.reward_op_negat, feed_dict={self.generator_obs_ph: proper_shape(a_obs_buf),
                                                                     self.generator_acs_ph: proper_shape(a_act_buf)})

    def get_positive_reward_buf(self,session, a_obs_buf, a_act_buf):#get_reward
        return session.run(self.reward_op_posit, feed_dict={self.generator_obs_ph: proper_shape(a_obs_buf),
                                                                     self.generator_acs_ph: proper_shape(a_act_buf)})

    def get_airl_reward_buf(self,session, a_obs_buf, a_act_buf):#get_reward
        return session.run(self.reward_op_airl, feed_dict={self.generator_obs_ph: proper_shape(a_obs_buf),
                                                                     self.generator_acs_ph: proper_shape(a_act_buf)})

    def get_reward(self,session, a_obs_buf, a_act_buf):
        return session.run(self.reward_op, feed_dict={self.generator_obs_ph: proper_shape(a_obs_buf),
                                                                     self.generator_acs_ph: proper_shape(a_act_buf)})
        

    def build_ph(self):#make placeholder shape(None,obs.shape)
        self.generator_obs_ph = tf.placeholder(tf.float32, combined_shape(None,self.observation_shape), name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, combined_shape(None,self.actions_shape), name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, combined_shape(None,self.observation_shape), name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, combined_shape(None,self.actions_shape), name="expert_actions_ph")


    def build_graph(self, obs_ph, acs_ph, reuse=False):#2層のtanhと全結合層
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            
            obs = obs_ph#!とりあえず標準化なし
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits#確率
    
        def proper_shape(arr):#!input numpyarray
            if arr.ndim==1:
                return np.reshape(arr,(arr.shape[0],1))
            return arr
