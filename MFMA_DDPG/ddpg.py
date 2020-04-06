import io
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from .model import Actor,Critic



class DDPG(object):
    def __init__(self,agent_args):
        self.actor_lr = agent_args['actor_lr']
        self.critic_lr = agent_args['critic_lr']
        self.lr_decay = agent_args['lr_decay']
        self.l2_critic = agent_args['l2_critic']
        self.discount = agent_args['discount']
        self.tau = agent_args['tau']
        self.with_cuda = agent_args['with_cuda']
        self.buffer_size = int(agent_args['buffer_size'])
        
    def setup(self, nb_pos,nb_laser, nb_actions,model_args):
        self.lr_coef = 1
        actor  = Actor (nb_pos,nb_laser, nb_actions, hidden1 = model_args['hidden1'], hidden2 = model_args['hidden2'] , layer_norm = model_args['layer_norm'])
        critic = Critic(nb_pos,nb_laser, nb_actions,hidden1 = model_args['hidden1'], hidden2 = model_args['hidden2'] , layer_norm = model_args['layer_norm'])
        self.actor         = copy.deepcopy(actor)
        self.actor_target  = copy.deepcopy(actor)
        self.critic        = copy.deepcopy(critic)
        self.critic_target = copy.deepcopy(critic)
        
        
        if self.with_cuda:
            for net in (self.actor, self.actor_target, self.critic, self.critic_target):
                if net is not None:
                    net.cuda()
        
        p_groups = [{'params': [param,],
                     'weight_decay': self.l2_critic if ('weight' in name) and ('LN' not in name) else 0
                    } for name,param in self.critic.named_parameters() ]
        self.critic_optim  = Adam(params = p_groups, lr=self.critic_lr, weight_decay = self.l2_critic)
        self.actor_optim  = Adam(self.actor.parameters(), lr=self.actor_lr)
        
    def update_critic(self, batch):
        tensor_obs0 = batch['obs0']
        tensor_obs1 = batch['obs1']
        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target(tensor_obs1[0], tensor_obs1[1],self.actor_target(tensor_obs1[0], tensor_obs1[1]))
            target_q_batch = batch['rewards'] + self.discount*(1-batch['terminals1'])*next_q_values
        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic(tensor_obs0[0],tensor_obs0[1], batch['actions'])
        value_loss = nn.functional.mse_loss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()
        return value_loss.item()
        
    def update_actor(self, batch):
        assert batch is not None  
        tensor_obs0 = batch['obs0']
        # Actor update
        self.actor.zero_grad()
        policy_loss = -self.critic(tensor_obs0[0],tensor_obs0[1],self.actor(tensor_obs0[0],tensor_obs0[1]))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()  
        return policy_loss.item()

    def update_critic_target(self,soft_update = True):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau \
                                    if soft_update else param.data)

    def update_actor_target(self,soft_update = True):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau \
                                    if soft_update else param.data)
                                    
    def apply_lr_decay(self):
        if self.lr_decay > 0:
            self.lr_coef = self.lr_decay*self.lr_coef/(self.lr_coef+self.lr_decay)
            for (opt,base_lr) in ((self.actor_optim,self.actor_lr),(self.critic_optim,self.critic_lr)):
                for group in opt.param_groups:
                    group['lr'] = base_lr * self.lr_coef
            
    def load_weights(self, model_dir): 
        self.actor  = torch.load('{}/actor.pkl'.format(model_dir) )
        self.critic = torch.load('{}/critic.pkl'.format(model_dir))
            
    def save_model(self, model_dir):
        torch.save(self.actor ,'{}/actor.pkl'.format(model_dir) )
        torch.save(self.critic,'{}/critic.pkl'.format(model_dir))
            
    def get_actor_buffer(self):
        actor_buffer = io.BytesIO()
        torch.save(self.actor, actor_buffer)
        return actor_buffer