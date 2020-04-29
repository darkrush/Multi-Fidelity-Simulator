import gym 
import numpy as np
import copy
from .random_map import random_fence
from .core import Action,World
from .utils import temp_agent_prop

def sample_in_radio(R,center_coord = [0,0]):
    theta = np.random.rand()*2*np.pi
    r = R*np.sqrt(np.random.rand())
    return r*np.cos(theta)+center_coord[0], r*np.sin(theta) + center_coord[1]

def sample_from_placeable(placeable_list,car_R):
    room_id = np.random.randint(len(placeable_list))
    x = placeable_list[room_id][0] + car_R \
        + (placeable_list[room_id][2] - 2*car_R) * np.random.rand()
    y = placeable_list[room_id][1] + car_R \
        + (placeable_list[room_id][3] - 2*car_R) * np.random.rand()
    return x,y

def check_in_placeable(placeable_list,car_R, coord):
    x = coord[0]
    y = coord[1]
    for room in placeable_list:
        x_in = room[0]+car_R< x < room[0]+room[2]-car_R
        y_in = room[1]+car_R< y < room[1]+room[3]-car_R
        if x_in and y_in:
            return True
    return False

def near_placeable_sample(placeable_list,
                          input_all_coord, sample_flag,
                          target_all_coord, target_flag,
                          car_R, near_dist, dead_count = 10000):
    i_coord_list = copy.deepcopy(input_all_coord)
    t_coord_list = copy.deepcopy(target_all_coord)
    all_number = len(i_coord_list)
    for idx in range(all_number):
        if not sample_flag[idx] and not target_flag[idx]:
            continue
        if not sample_flag[idx] and target_flag[idx]:
            while True:
                x,y = sample_in_radio(near_dist,i_coord_list[idx])
                if check_in_placeable(placeable_list,car_R, [x,y]):
                    break
            t_coord_list[idx] = [x,y]

        if sample_flag[idx] :
            if target_flag[idx]:
                t_coord_list[idx] = sample_from_placeable(placeable_list,car_R)
            while True:
                x,y = sample_in_radio(near_dist,t_coord_list[idx])
                if check_in_placeable(placeable_list,car_R, [x,y]):
                    break
            i_coord_list[idx] = [x,y]


        failed = False
        for all_coord in (i_coord_list, t_coord_list):
            for (pos_id_1) in range(all_number):    
                for pos_id_2 in range(pos_id_1+1,all_number):
                    dist_squre = (all_coord[pos_id_1][0]-all_coord[pos_id_2][0])**2 \
                                +(all_coord[pos_id_1][1]-all_coord[pos_id_2][1])**2
                    if dist_squre<(2*car_R)**2:
                        failed = True
                        break
                if failed:
                    break
            if failed:
                break
        if not failed :
            break
        dead_count = dead_count - 1
        if dead_count == 0:
            return None
    return i_coord_list,t_coord_list

    

def placeable_sample(placeable_list, input_all_coord,
                     sample_flag, car_R, dead_count = 1000):
    all_coord = copy.deepcopy(input_all_coord)
    sample_id_list = [ sample_id for (sample_id,flag) in enumerate(sample_flag) if flag is True]
    all_number = len(all_coord)
    while dead_count>0:
        for sample_id in sample_id_list:
            all_coord[sample_id] = sample_from_placeable(placeable_list,car_R)

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


def random_agent(placeable_list, agent_number, agent_prop = None):
    if agent_prop is None:
        Agent_prop = temp_agent_prop()
    else:
        Agent_prop = agent_prop
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
                      agent_number = 3,
                      near_dist = -1.0,
                      crash_reward = -10,
                      reach_reward =  10,
                      potential = 1.0,
                      time_penalty = -0.1,
                      dt = 0.1,
                      nb_step = 1,
                      agent_prop = None):
        if agent_prop is None:
            self.agent_prop = temp_agent_prop()
        else:
            self.agent_prop = agent_prop
        self.map_W = map_W
        self.map_H = map_H
        self.near_dist = near_dist
        self.room_number = room_number
        self.door_width = door_width
        self.half_wall_width = half_wall_width
        self.agent_number = agent_number
        self.crash_reward = crash_reward
        self.reach_reward = reach_reward
        self.potential = potential
        self.time_penalty = time_penalty
        self.R_safe = self.agent_prop['R_safe']
        self.world = None
        low_array = np.array([0.0 for _ in range(self.agent_prop['N_laser'])]+[0,0,-1,-1,0,0])
        low_array = np.vstack([low_array for _ in range(self.agent_number)]) 
        high_array = np.array([self.agent_prop['R_laser'] for _ in range(self.agent_prop['N_laser'])]+[map_W,map_H,1,1,map_W,map_H])
        high_array = np.vstack([high_array for _ in range(self.agent_number)]) 
        self.observation_space = gym.spaces.Box(low = low_array,high = high_array)
        self.action_space = gym.spaces.Box(-1,1,(self.agent_number,2,))

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
        agent_dict = random_agent(placeable_list, self.agent_number,self.agent_prop )
        if self.world is None:
            self.world = World(agent_dict,fence_dict,dt,nb_step)
        else:
            self.world.setup(agent_dict,fence_dict)
        new_state = self.world.get_state()
        new_state,_,_,_ = self._random_reset(new_state,all_reset=True,near_dist = self.near_dist)
        self.world.set_state(new_state)
        obs = self.world.get_obs()
        self.last_state = new_state
        obs_array = np.vstack([np.hstack([obs_idx.laser_data,obs_idx.pos]) for obs_idx in obs])
        return obs_array

    def step(self,action):
        assert self.world is not None
        action_list = [Action(a[0],a[1]) for a in action]
        self.world.set_action(action_list)
        self.world.step()
        obs = self.world.get_obs()
        
        obs_array = np.vstack([np.hstack([obs_idx.laser_data,obs_idx.pos]) for obs_idx in obs])
        new_state = self.world.get_state()
        reward = self._calc_reward(new_state,self.last_state)
        new_state,dead,crash,reach = self._random_reset(new_state,all_reset=False,near_dist = self.near_dist)
        self.world.set_state(new_state)
        done = False
        info = {'dead':dead,'crash':crash,'reach':reach}
        self.last_state = copy.deepcopy(new_state)
        return obs_array,reward,done,info
    
    def _calc_reward(self,new_state,old_state):
        all_reward = []
        for ns,os in zip(new_state,old_state):
            old_dis = ((os.x-os.target_x)**2+(os.y-os.target_y)**2)**0.5
            new_dis = ((ns.x-ns.target_x)**2+(ns.y-ns.target_y)**2)**0.5
            potential_reward = (old_dis-new_dis)*self.potential
            crash = self.crash_reward if ns.crash else 0
            reach = self.reach_reward if ns.reach else 0
            potential_reward = potential_reward if not (crash or reach) else 0

            reward = crash + reach + potential_reward + self.time_penalty
            all_reward.append(reward)
        return all_reward

    def _random_reset(self,state_list, all_reset = False, near_dist = -1):
        coord_list = [[state.x,state.y] for state in state_list]
        reset_flag = [state.crash or all_reset for state in state_list]
        target_coord_list = [[state.target_x,state.target_y] for state in state_list]
        target_reset_flag = [state.reach or all_reset for state in state_list]
        if near_dist <0:
            new_coord_list = placeable_sample(self.placeable_list,coord_list,reset_flag,self.R_safe)
            new_target_list = placeable_sample(self.placeable_list,target_coord_list,target_reset_flag,self.R_safe)
        else:
            new_coord_list,new_target_list = near_placeable_sample(self.placeable_list,
                                                                   coord_list, reset_flag,
                                                                   target_coord_list, target_reset_flag,
                                                                   self.R_safe,near_dist)
        dead = [reach or crash for (reach,crash) in zip(reset_flag, target_reset_flag)]
        crash = [state.crash for state in state_list]
        reach = [state.reach for state in state_list]
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
        return state_list,dead,crash,reach

    def close(self):
        pass

    def seed(self):
        pass