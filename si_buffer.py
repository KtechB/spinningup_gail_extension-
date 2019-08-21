import numpy as np
import tensorflow as tf
from collections import deque
from operator import itemgetter

def proper_shape(arr):#!input numpyarray
    if arr.ndim==1:
        return np.reshape(arr,(arr.shape[0],1))
    return arr
class SIBuffer():
    def __init__(self,obs_space, act_space, expert_obs,expert_act,trj_num = 20,max_size= None,js_smooth_num= 20,sm_r_ratio =0):
        if max_size == None:
            max_size = expert_obs.shape[0]#max_sizeに指定がなければデモの長さ
        assert expert_obs.shape[0]<= max_size,  'demosize[{0}] < max_size[{1}]'.format( expert_obs.shape[0], max_size)
        assert sm_r_ratio>= 0 and sm_r_ratio<1
        self.trj_buf = deque(maxlen = trj_num)
        self.obs_space = obs_space#must be (n,)or()
        self.act_space = act_space
        self.obs_def_shape = 1  if self.obs_space ==() else self.obs_space[0]#1次元にも対応
        self.act_def_shape = 1  if self.act_space ==() else self.act_space[0]
        self.expert_obs = expert_obs.reshape(-1,self.obs_def_shape)
        self.expert_act = expert_act.reshape(-1,self.act_def_shape)
        self.main_obs_buf = expert_obs.reshape(-1,self.obs_def_shape)
        self.main_act_buf = expert_act.reshape(-1,self.act_def_shape)
        self.merge_ratio = 0

        self.js_randam_agent =0

        self.js_buf = deque(maxlen = int(js_smooth_num))
        self.smoothed_js=None
        self.js_ratio_with_random =None

        self.r_buf = deque(maxlen = int(js_smooth_num))
        self.smoothed_r=None

        self.sm_r_ratio = sm_r_ratio
        
        self.ptr, self.max_size,self.buffer_r_average = 0, max_size,0

    def store(self,obs_trj, act_trj, sum_reward):
        
        
        rewards = [x[2] for x in self.trj_buf]
        self.buffer_r_average = (sum(rewards)/len(rewards)) if len(rewards)>0 else 0
        if not(self.smoothed_r is None):#until self.smoothed_r is curriculate ,dont store
            sum_reward  = (1-self.sm_r_ratio)*sum_reward+ self.sm_r_ratio*self.smoothed_r

            if sum_reward >= self.buffer_r_average or len(rewards)==0 :#!rewardがbufの平均以上なら追加 <= から<に変更
                iteration_trj = [obs_trj, act_trj , sum_reward]
                self.trj_buf.append(iteration_trj)
    
    def store_everytrj(self,obs_trj, act_trj, sum_reward):
        iteration_trj = [obs_trj, act_trj , sum_reward]
        self.trj_buf.append(iteration_trj)
   
    def get_obs_trj(self):
        obs_buf = np.empty((0,self.obs_def_shape))#配列の初期化
        for i in range(len(self.trj_buf)):
            obs_buf= np.append(obs_buf,proper_shape(self.trj_buf[i][0]),axis = 0)#obsを連結 axisを指定しないと１次元配列になってしまうことに注意
        self.obs_buf = obs_buf
        return obs_buf
    
    def get_act_trj(self):
        act_buf = np.empty((0,self.act_def_shape))#配列の初期化　２次元以上のときはここもいじるべし
        for i in range(len(self.trj_buf)):
            
            act_buf= np.append(act_buf,proper_shape(self.trj_buf[i][1]),axis =0)#obsを連結
        self.act_buf = act_buf
        return act_buf
    
    def get_random_batch(self,batch_size, shuffle =True):
        obs, act = self.main_obs_buf,self.main_act_buf
        obs_len =obs.shape[0]
        if batch_size <= obs_len:# 'batchsize bigger than buf_size'
            obs_batch =obs[np.random.choice(obs.shape[0], batch_size, replace=False), :]
            act_batch =act[np.random.choice(act.shape[0], batch_size, replace=False), :]
        else:
            repete_num=batch_size//obs_len
            obs_batch =np.zeros((batch_size,obs.shape[1]))
            act_batch =np.zeros((batch_size,obs.shape[1]))

            for n in range(repete_num):
                obs_batch[n*obs_len:(n+1)*obs_len] = obs
                act_batch[n*obs_len:(n+1)*obs_len] = act

            obs_batch[(repete_num)*obs_len:]=obs[np.random.choice(obs.shape[0], batch_size%obs_len, replace=False), :]
            act_batch[(repete_num)*obs_len:]=act[np.random.choice(act.shape[0], batch_size%obs_len, replace=False), :]

        return obs_batch, act_batch

    
    def update_main_buf(self,ratio_update =0.02,update_type = 'random'):
        if update_type == 'random':
            self.update_main_buf_random(ratio_update)
        elif update_type == 'merge':
            self.update_main_buf_by_merge(ratio_update)
        else:
            raise 'Error:update_type must be random or merge!'

    def update_main_buf_random(self, ratio_update = 0.02):
        num_of_update = int(self.max_size*ratio_update)
        obs_buf = self.get_obs_trj()
        act_buf = self.get_act_trj()
        assert obs_buf.shape[0] == act_buf.shape[0]
        if self.ptr <= obs_buf.shape[0]-1-num_of_update:#バッファからはみ出た時点でstr =0
            i =self.ptr
            self.ptr +=num_of_update
        else:
            self.ptr =0
            i= self.ptr
        
        self.main_obs_buf=np.append(self.main_obs_buf, obs_buf[i:i+num_of_update],axis=0)
        self.main_act_buf=np.append(self.main_act_buf, act_buf[i:i+num_of_update],axis=0)
        delete_num = self.main_obs_buf.shape[0]-self.max_size
        if delete_num > 0 :
            self.main_obs_buf = np.delete(self.main_obs_buf, [i for i in range(delete_num)], axis =0)
            self.main_act_buf = np.delete(self.main_act_buf, [i for i in range(delete_num)], axis =0)
            assert self.main_obs_buf.shape[0]== self.max_size
    
    def change_main_buf(self):
        self.sort_with_reward()
        obs_buf = self.get_obs_trj()
        act_buf = self.get_act_trj()
        assert obs_buf.shape[0] == act_buf.shape[0]
        self.main_obs_buf = obs_buf[-(self.max_size):]#maxsize個抜き出して代入
        self.main_act_buf = act_buf[-(self.max_size):]

    def update_main_buf_by_merge(self,ratio_update= 0.02):
        self.sort_with_reward()
        self.merge_ratio += ratio_update
        if self.merge_ratio >1.0:
            self.merge_ratio =1.0
        merge_ratio = self.merge_ratio
        assert merge_ratio <=1 and merge_ratio>= 0

        obs_buf = self.get_obs_trj()
        act_buf = self.get_act_trj()
        assert obs_buf.shape[0] == act_buf.shape[0]
        self.main_obs_buf[:int(self.max_size*(1-merge_ratio))]=self.expert_obs[:int(self.max_size*(1-merge_ratio))]
        self.main_act_buf[:int(self.max_size*(1-merge_ratio))]=self.expert_act[:int(self.max_size*(1-merge_ratio))]

        self.main_obs_buf[int(self.max_size*(1-merge_ratio)):self.max_size]=obs_buf[:(self.max_size-int(self.max_size*(1-merge_ratio)))]
        self.main_act_buf[int(self.max_size*(1-merge_ratio)):self.max_size]=act_buf[:(self.max_size-int(self.max_size*(1-merge_ratio)))]

    def update_main_buf_by_merge_latest(self,ratio_update= 0.02):
        self.merge_ratio += ratio_update
        if self.merge_ratio >1.0:
            self.merge_ratio =1.0
        merge_ratio = self.merge_ratio
        assert merge_ratio <=1 and merge_ratio>= 0

        obs_buf = self.get_obs_trj()
        act_buf = self.get_act_trj()
        assert obs_buf.shape[0] == act_buf.shape[0]
        self.main_obs_buf[:int(self.max_size*(1-merge_ratio))]=self.expert_obs[:int(self.max_size*(1-merge_ratio))]
        self.main_act_buf[:int(self.max_size*(1-merge_ratio))]=self.expert_act[:int(self.max_size*(1-merge_ratio))]

        self.main_obs_buf[int(self.max_size*(1-merge_ratio)):self.max_size]=obs_buf[-(self.max_size-int(self.max_size*(1-merge_ratio))):]
        self.main_act_buf[int(self.max_size*(1-merge_ratio)):self.max_size]=act_buf[-(self.max_size-int(self.max_size*(1-merge_ratio))):]
    
        
    def get_main_buf(self):
        return self.main_obs_buf, self.main_act_buf
    
    def store_js(self,js):
        self.js_buf.append(js)
        if len(self.js_buf)==self.js_buf.maxlen:
            if self.js_randam_agent ==0:
                self.js_randam_agent =sum(self.js_buf)/len(self.js_buf)

            self.js_smoothed  = sum(self.js_buf)/len(self.js_buf)
            self.js_ratio_with_random = self.js_smoothed/self.js_randam_agent
        else:
            self.js_smoothed=None

    def store_r(self,r):
        self.r_buf.append(r)
        if len(self.r_buf)==self.r_buf.maxlen:
            self.smoothed_r  = sum(self.r_buf)/len(self.r_buf)
        else:
            self.smoothed_r=None

    def sort_with_reward(self):
        self.trj_buf=sorted(self.trj_buf,key =itemgetter(2))

    