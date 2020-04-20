from HF_Sim.random_map import random_room,place_door,wall2fence
from HF_Sim.simulator import random_agent, Full_env
from HF_Sim.core import Action,World
from HF_Sim.utils import naive_policy
import time
import numpy as np
import math


policy_args={}
policy_args['max_phi'] = math.pi/6.0
policy_args['l'] = 0.3
policy_args['dist'] = 0.1

n_policy = naive_policy(policy_args)


half_wall_width = 0.05
car_R = 0.2
door_width = 0.8
room_number = 5
agent_number = 3

Env = Full_env()
obs = Env.reset()
action = Action()
action.ctrl_phi = 1.0
action.ctrl_vel = 1.0
while True:
    action_list = n_policy.inference(obs,[])
    #action.ctrl_phi = -action.ctrl_phi
    
    obs,reward,done,info = Env.step(action_list)
    print('************')
    print(obs)
    print(reward)
    Env.render()


Full_env