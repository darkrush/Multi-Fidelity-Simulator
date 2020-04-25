import gym 
import numpy as np
import copy
from .random_map import random_fence
from .core import Action,World
from .utils import temp_agent_prop



def placeable_sample(placeable_list, input_all_coord,
                     sample_flag, car_R, dead_count = 1000):
    all_coord = copy.deepcopy(input_all_coord)
    sample_id_list = [ sample_id for (sample_id,flag) in enumerate(sample_flag) if flag is True]
    all_number = len(all_coord)
    while dead_count>0:
        for sample_id in sample_id_list:
            room_id = np.random.randint(len(placeable_list))
            x = placeable_list[room_id][0] + car_R \
              + (placeable_list[room_id][2] - 2*car_R) * np.random.rand()
            y = placeable_list[room_id][1] + car_R \
              + (placeable_list[room_id][3] - 2*car_R) * np.random.rand()
            all_coord[sample_id][0] = x
            all_coord[sample_id][1] = y

        failed = False
        for (pos_id_1) in range(all_number):
            for pos_id_2 in range(pos_id_1+1,all_number):
                dist_squre = (all_coord[pos_id_1][0]-all_coord[pos_id_2][0])**2 \
                            +(all_coord[pos_id_1][1]-all_coord[pos_id_2][1])**2
                if dist_squre<(2*car_R)**2:
                    failed = True
                    break
            if failed :
                break
        if not failed:
            break
        dead_count = dead_count - 1
        if dead_count == 0:
            return None
    return all_coord


def random_agent(placeable_list, agent_number):
    Agent_prop = temp_agent_prop()
    R_Safe = Agent_prop['R_safe']
    dummy_coord = [[0,0] for _ in range(agent_number)]
    sample_flag = [True for _ in range(agent_number)]
    agent_init_coord = placeable_sample(placeable_list,dummy_coord,sample_flag,R_Safe)
    agent_target_coord = placeable_sample(placeable_list,dummy_coord,sample_flag,R_Safe)
    main_group = []
    for agent_id in range(agent_number):
        agent_prop = Agent_prop.copy()
        agent_prop['init_x'] = agent_init_coord[agent_id][0]
        agent_prop['init_y'] = agent_init_coord[agent_id][1]
        agent_prop['init_theta'] = np.random.rand()*6.28
        agent_prop['init_vel_b'] = 0
        agent_prop['init_phi'] = 0
        agent_prop['init_target_x'] = agent_target_coord[agent_id][0]
        agent_prop['init_target_y'] = agent_target_coord[agent_id][1]
        main_group.append(agent_prop)

    Agent_list = {'main_group':main_group}
    return Agent_list

class Full_env(gym.Env):
    def __init__(self,map_W = 8.0, map_H = 8.0,
                      room_number = 5,
                      door_width = 0.8,
                      half_wall_width = 0.05,
                      agent_number = 3):
        self.temp_agent_prop = temp_agent_prop()
        self.map_W = map_W
        self.map_H = map_H
        self.room_number = room_number
        self.door_width = door_width
        self.half_wall_width = half_wall_width
        self.agent_number = agent_number
        self.R_safe = self.temp_agent_prop['R_safe']
        self.world = None

    def render(self):
        assert self.world is not None
        self.world.render()

    def reset(self):
        result = random_fence(map_W = self.map_W,
                              map_H = self.map_H,
                              half_wall_width = self.half_wall_width,
                              car_R = self.R_safe,
                              door_width = self.door_width,
                              room_number = self.room_number)
        fence_dict, placeable_list = result
        self.placeable_list = placeable_list
        agent_dict = random_agent(placeable_list, self.agent_number)
        if self.world is None:
            self.world = World(agent_dict,fence_dict)
        else:
            self.world.setup(agent_dict,fence_dict)
        new_state = self.world.get_state()
        new_state,dead = self._random_reset(new_state,all_reset=True)
        self.world.set_state(new_state)
        obs = self.world.get_obs()
        self.last_state = new_state
        return obs

    def step(self,action):
        assert self.world is not None
        action_list = [Action(a[0],a[1]) for a in action]
        self.world.set_action(action_list)
        self.world.step()
        obs = self.world.get_obs()
        new_state = self.world.get_state()
        reward = self._calc_reward(new_state,self.last_state)
        new_state,dead = self._random_reset(new_state)
        self.world.set_state(new_state)
        done = False
        info = {'dead':dead}
        self.last_state = new_state
        return obs,reward,done,info
    
    def _calc_reward(self,new_state,old_state):
        all_reward = []
        for ns,os in zip(new_state,old_state):
            old_dis = ((os.x-os.target_x)**2+(os.y-os.target_y)**2)**0.5
            new_dis = ((ns.x-ns.target_x)**2+(ns.y-ns.target_y)**2)**0.5
            potential_reward = old_dis-new_dis
            crash = -10 if ns.crash else 0
            reach = 10 if ns.reach else 0
            reward = crash + reach + potential_reward
            all_reward.append(reward)
        return all_reward

    def _random_reset(self,state_list, all_reset = False):
        coord_list = [[state.x,state.y] for state in state_list]
        reset_flag = [state.crash or all_reset for state in state_list]
        target_coord_list = [[state.target_x,state.target_y] for state in state_list]
        target_reset_flag = [state.reach or all_reset for state in state_list]
        new_coord_list = placeable_sample(self.placeable_list,coord_list,reset_flag,self.R_safe)
        new_target_list = placeable_sample(self.placeable_list,target_coord_list,target_reset_flag,self.R_safe)
        dead = [reach or crash for (reach,crash) in zip(reset_flag, target_reset_flag)]
        for idx,state in enumerate(state_list):
            if reset_flag[idx]:
                state_list[idx].theta = np.random.uniform(0,3.1415926*2)
            state_list[idx].reach = False
            state_list[idx].crash = False
            state_list[idx].movable = True
            state_list[idx].x = new_coord_list[idx][0]
            state_list[idx].y = new_coord_list[idx][1]
            state_list[idx].target_x = new_target_list[idx][0]
            state_list[idx].target_y = new_target_list[idx][1]
        return state_list,dead

    def close(self):
        pass

    def seed(self):
        pass