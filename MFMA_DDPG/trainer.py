import numpy as np
import os
import pickle
import torch
import gym
from copy import deepcopy
from .arguments import Singleton_arger
from .logger import Singleton_logger
from .ddpg import DDPG
import sys
sys.path.append("..")
from MF_env import policy

class DDPG_trainer(object):
    def __init__(self):
        train_args = Singleton_arger()['train']
        self.nb_epoch = train_args['nb_epoch']
        self.nb_cycles_per_epoch = train_args['nb_cycles_per_epoch']
        self.nb_rollout_steps = train_args['nb_rollout_steps']
        self.nb_train_steps = train_args['nb_train_steps']
        self.nb_warmup_steps = train_args['nb_warmup_steps']
        self.train_mode = train_args['train_mode']
        self.batch_size = train_args['batch_size']

    def setup(self,env_instance,eval_env_instance,agent,memory,ctrl_fps):
        main_args = Singleton_arger()['main']
        Singleton_logger.setup(main_args['result_dir'])
        self.env = env_instance
        self.eval_env = eval_env_instance
        self.agent = agent
        self.memory = memory
        self.ctrl_fps = ctrl_fps
        self.result_dir = main_args['result_dir']
    
    def train(self):
        total_cycle=0
        for epoch in range(self.nb_epoch):
            for cycle in range(self.nb_cycles_per_epoch):
                total_cycle+=1
                self.env.reset_rollout()
                rollout_policy = policy.NN_policy(self.agent.actor,10.0/(total_cycle+10.0))
                self.env.rollout(rollout_policy.inference, self.ctrl_fps)
                trajectoy = self.env.get_trajectoy()
                results = self.env.get_result()
                for key,value in results.items():
                    Singleton_logger.add_scalar('train'+key,value,total_cycle)
                for traj in (trajectoy):
                    for idx_agent in range(len(traj['done'])):
                        self.memory.append( [traj['obs'][idx_agent].pos,traj['obs'][idx_agent].laser_data], 
                                            [traj['action'][idx_agent].ctrl_vel,traj['action'][idx_agent].ctrl_phi],
                                            traj['reward'][idx_agent],
                                            [traj['obs_next'][idx_agent].pos,traj['obs_next'][idx_agent].laser_data],
                                            traj['done'][idx_agent])
                cl,al = self.apply_train()
                Singleton_logger.add_scalar('critic_loss',cl,total_cycle)
                Singleton_logger.add_scalar('actor_loss',al,total_cycle)

                
            self.eval_env.reset_rollout()
            rollout_policy = policy.NN_policy(self.agent.actor,0)
            self.eval_env.rollout(rollout_policy.inference, self.ctrl_fps)
            results = self.eval_env.get_result()
            print(results)
            for key,value in results.items():
                Singleton_logger.add_scalar('eval'+key,value,total_cycle)
            Singleton_logger.save_dict()
            self.agent.save_model(self.result_dir)

    def apply_train(self):
        #update agent for nb_train_steps times
        cl_list = []
        al_list = []
        if self.train_mode == 0:
            for t_train in range(self.nb_train_steps):
                cl = self.agent.update_critic(self.memory.sample(self.batch_size))
                al = self.agent.update_actor(self.memory.sample(self.batch_size))
                self.agent.update_critic_target()
                self.agent.update_actor_target()
                cl_list.append(cl)
                al_list.append(al)
        elif self.train_mode == 1:
            for t_train in range(self.nb_train_steps):
                cl = self.agent.update_critic(self.memory.sample(self.batch_size))
                cl_list.append(cl)
                al = self.agent.update_actor(self.memory.sample(self.batch_size))
                al_list.append(al)
            self.agent.update_critic_target(soft_update = False)
            self.agent.update_actor_target (soft_update = False)
        return np.mean(cl_list),np.mean(al_list)
                