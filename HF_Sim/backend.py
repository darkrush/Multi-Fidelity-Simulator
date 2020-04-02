import sys 
from . import basic
from . import core
import time

class MSE_backend(object):
    def __init__(self,scenario,window_scale = 1.0, dt = 0.1):
        self.agent_groups = scenario['agent_groups']
        self.fence_list = scenario['fence_list']
        self.cfg = {'dt': dt,'window_scale':window_scale}
        self.use_gui = scenario['common']['use_gui']
        
        self.world = core.World(self.agent_groups,self.fence_list,self.cfg)
    
    def step(self,step_number = 1):
        render_frams = step_number if step_number <= 40 else int(step_number / 10)
        total_frams = step_number
        while total_frams>=0:
            self.world.step()
            if self.use_gui :
                self.world.render(time = '%.2f'%self.world.get_total_time())
            total_frams-=render_frams


    def get_state(self):
        return [self.world.total_time,self.world.get_state()]

    def set_state(self,state,enable_list = None,reset = False):
        if reset:
            self.world.reset()
        if enable_list is None:
            enable_list = [True]* len(state)
        self.world.set_state(enable_list,state)
        

    def get_obs(self):
        return self.world.get_obs()

    def set_action(self,actions,enable_list= None):
        if enable_list is None:
            enable_list = [True]* len(actions)
        self.world.set_action(enable_list,actions)