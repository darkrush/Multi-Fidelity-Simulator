import numpy as np
import math

def temp_agent_prop():
    basic_agent_prop = {}
    basic_agent_prop['R_safe'] = 0.20
    basic_agent_prop['R_reach']= 0.1
    basic_agent_prop['L_car']  = 0.30
    basic_agent_prop['W_car']  = 0.20
    basic_agent_prop['L_axis'] = 0.20
    basic_agent_prop['R_laser']= 3.0
    basic_agent_prop['N_laser']= 32
    basic_agent_prop['K_vel']  = 0.8266     # coefficient of back whell velocity control
    basic_agent_prop['K_phi']   = 0.2983   # coefficient of front wheel deflection control 
    basic_agent_prop['init_movable'] =  True
    basic_agent_prop['init_enable'] =  True
    return basic_agent_prop


def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    #r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return (r, g, b)

class naive_policy(object):
    def __init__(self,policy_args):
        self.max_phi = policy_args['max_phi']
        self.l = policy_args['l']
        self.dist = policy_args['dist']
        self.min_r = self.l/np.tan(self.max_phi)
        self.right_o = np.array([self.min_r,0.0])
        self.left_o = np.array([-self.min_r,0.0])
    
    def inference(self,obs_list,new_state):
        action_list = []
        for obs in obs_list:
            print(obs)
            obs = obs[-6:]
            xt = obs[4] - obs[0]
            yt = obs[5] - obs[1]
            xt,yt = (xt*obs[2]+yt*obs[3],yt*obs[2]-xt*obs[3])
    
            if abs(yt) < self.dist:
                vel = np.sign(xt)
                phi = 0
            else:
                in_min_r = (xt**2+(abs(yt)-self.min_r)**2)< self.min_r**2
                vel = -1 if np.bitwise_xor(in_min_r,xt<0) else 1
                phi = -1 if np.bitwise_xor(in_min_r,yt<0) else 1
            action_list.append([vel,phi])
        return action_list
