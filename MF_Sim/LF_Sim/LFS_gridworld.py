#-*- coding: UTF-8 -*-
import numpy as np
import math
import random
import gym
from gym import spaces
import copy
import sys
sys.path.append("..")
from multiagent_particle_envs.multiagent.multi_discrete import MultiDiscrete

class Agent(object):
    def __init__(self):
        #position
        self.pos = Coordinate()
        #目标点
        self.goal = Coordinate()
        #action
        self.act = None
        #movable
        self.movable = True
        #reach
        self.reach = False
        #朝向
        self.direct = -1
        #crash
        self.crash = False
        #速度
        self.vel = 1.47
        #转向半径
        self.r = 1.848
        #存储上一个state
        self.s_buffer = Coordinate()
        #长宽
        self.L_car = 0.8
        self.W_car = 0.4
        self.color = [255,0,0]
        #crash与reach的reward标志




#坐标类
class Coordinate(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

class ob_space(object):
    def __init__(self,x=0,y=0,z=0):
        self.shape = (x,y,z)

class world(object):
    def __init__(self, length=0, width=0, is_random=False):
        self.res = 1 #1米每格
        #地图长宽
        self.len = length
        self.wid = width
        #起点终点
        self.goal_list = []
        self.goal_map = np.zeros((self.len, self.wid), dtype = np.int_)

        self.theta = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]        
        self.agents = []

        self.viewer = None
        self.agent_geom_list = None
        self.grid_geom_list = None
        self.action_space = []
        self.observation_space = []
        #self.direct_space = 2
        self.cam_range = 15
        self.rewcnt = 0
        self.crashcnt = 0
        self.pcnt = 0
        self.mcnt = 0
        self.rcnt = 0
        if is_random:
            self.random_fill(length, width)
        else:
            self.map = np.zeros((self.len, self.wid))
        #只包含障碍物的地图
        self.o_map = copy.deepcopy(self.map)

    def set_space(self):
        for agent in self.agents:
            #total_action_space = []
            # physical action space
            u_action_space = spaces.Discrete(6)
            # communication action space
            self.action_space.append(u_action_space)
            # observation space
            obs_dim = 6
            self.observation_space.append(ob_space(obs_dim,self.len,self.wid))
            #agent.action.c = np.zeros(self.world.dim_c)


    def set_world(self, res, len, wid, agent_list, goal_list, map):
        self.res = res
        self.len = len
        self.wid = wid
        self.agents = agent_list
        self.goal_list = goal_list
        self.map = map
        for agent in self.agents:
            agent.movable = True
            agent.reach = False
            agent.crash = False
            self.map[agent.pos.x][agent.pos.y] = 1

    def get_state(self):#[[x,y,direct,vel],[]...]
        state_list = []
        for agent in self.agents:
            state = [agent.pos.x, agent.pos.y, agent.direct, agent.vel]
            state_list.append(state)
        return state_list

    def set_action(self, action_list, vel_list):#[action1,action2...]
        cnt = 0
        for agent in self.agents:
            agent.act = action_list[cnt]
            agent.vel = vel_list[cnt]
            cnt = cnt + 1


    #判空
    def is_empty(self,x,y):
        return (not bool(self.o_map[x,y]))
    #随机填充
    def random_fill(self, len, wid):
        self.map = np.random.randint(0,1,(len,wid))

    #设置Agent
    def set_agent(self, Agent_list):
        self.agents = Agent_list

    #设置起点 传入一个元素为Coordinate类的list[C0,C1......] goalmap标注序号从1开始
    def set_start(self, start):
        cnt = 0
        for c in start:
            if c.x >= self.len or c.y >= self.wid or c.x<0 or c.y<0 :
                print("Wrong setting for start(",c.x,",",c.y,")")
                return(False)
            self.map[c.x, c.y] = 1
            self.agents[cnt].pos.x = c.x
            self.agents[cnt].pos.y = c.y
            cnt = cnt + 1
    #设置终点，格式同起点
    def set_goal(self, goal):
        self.goal_list = goal
        for (a,g) in zip(self.agents, self.goal_list):
            a.goal = g
        #cnt = 0
        #for c in goal:
        #    if c.x >= self.len or c.y >= self.wid or c.x<0 or c.y<0 :
        #        print("Wrong setting for goal(",c.x,",",c.y,")")
        #        return(False)
        #    cnt = cnt + 1

    #先把所有信息给好，再step
    def Astar(self):
        if len(self.agents) != 1:
            print("Num of agent is not 1 !")
            return None
        open_list = []
        close_list = []
        m_map = self.Astar_map()
        start_point = [self.Astar_trans(self.agents[0].pos.x, self.agents[0].pos.y, self.agents[0].direct), None, self.Astar_h(self.agents[0].pos.x, self.agents[0].pos.y), 0]
        open_list.append(start_point)
        goal = []
        track = []
        xytrack = []
        for dir in range(8):
            goal.append(self.Astar_trans(self.goal_list[0].x, self.goal_list[0].y, dir))

        while(len(open_list)>0):
            for g in goal:
                if self.check(g,open_list):
                    g_point = self.check(g,open_list)
                    tr = g_point
                    while(tr[1] is not None):
                        track.append(tr)
                        tr = self.check(tr[1],close_list)
                    for track_point in track:
                        [x,y,d] = self.Astar_inv(track_point[0])
                        xytrack.append([x,y,d])
                    return xytrack
            point = self.Astar_pick(open_list)
            open_list.remove(point)
            close_list.append(point)
            for cnt in range(8*self.len*self.wid):
                if m_map[point[0]][cnt] == 1:
                    [x,y,d] = self.Astar_inv(cnt)
                    if self.check(cnt,close_list):
                        continue
                    elif self.check(cnt,open_list):
                        check_point = self.check(cnt,open_list)
                        if point[3] +1 + self.Astar_h(x,y) < check_point[2]:
                            open_list.remove(check_point)
                            check_point[1] = point[0]
                            check_point[2] = point[3] +1 + self.Astar_h(x,y)
                            check_point[3] = point[3] + 1
                            open_list.append(check_point)
                    else:
                        open_list.append([cnt, point[0], point[3] +1 + self.Astar_h(x,y), point[3] + 1])


    def Astar_pick(self, open_list):
        p = open_list[0]
        f_min = open_list[0][2]
        for point in open_list:
            if f_min > point[2]:
                p = point
                f_min = point[2]
        return p

    def check(self,cnt,list):
        if len(list) == 0:
            return False
        for point in list:
            if point[0] == cnt:
                return point
        return False

        
    def Astar_h(self, x, y):
        return (abs(x-self.goal_list[0].x)+abs(y-self.goal_list[0].y))

    def Astar_trans(self,x,y,z):
        return (8*x*self.wid+8*y+z)
    
    def Astar_inv(self,x):
        return ([math.floor(x/(8*self.wid)), math.floor((x % (8*self.wid))/8), x % 8])

    def Astar_map(self):
        m_map = np.zeros((8*self.len*self.wid, 8*self.len*self.wid))
        for x in range(self.len):
            for y in range(self.wid):
                if self.map[x][y] == 1:
                    continue
                elif x == 0:
                    if y == self.wid-1:
                        self.Astar_fill(m_map,x,y,1,0)
                        self.Astar_fill(m_map,x,y,1,-1)
                        self.Astar_fill(m_map,x,y,0,-1)
                    elif y == 0:
                        self.Astar_fill(m_map,x,y,0,1)
                        self.Astar_fill(m_map,x,y,1,1)
                        self.Astar_fill(m_map,x,y,1,0)
                    else:
                        self.Astar_fill(m_map,x,y,1,1)
                        self.Astar_fill(m_map,x,y,1,0)
                        self.Astar_fill(m_map,x,y,1,-1)
                        self.Astar_fill(m_map,x,y,0,1)
                        self.Astar_fill(m_map,x,y,0,-1)
                elif x == self.len-1:
                    if y == 0:
                        self.Astar_fill(m_map,x,y,-1,1)
                        self.Astar_fill(m_map,x,y,-1,0)
                        self.Astar_fill(m_map,x,y,0,1)
                    elif y == self.wid-1:
                        self.Astar_fill(m_map,x,y,-1,0)
                        self.Astar_fill(m_map,x,y,-1,-1)
                        self.Astar_fill(m_map,x,y,0,-1)
                    else:
                        self.Astar_fill(m_map,x,y,-1,1)
                        self.Astar_fill(m_map,x,y,-1,0)
                        self.Astar_fill(m_map,x,y,-1,-1)
                        self.Astar_fill(m_map,x,y,0,1)
                        self.Astar_fill(m_map,x,y,0,-1)
                else:
                    if y == 0:
                        self.Astar_fill(m_map,x,y,-1,1)
                        self.Astar_fill(m_map,x,y,-1,0)
                        self.Astar_fill(m_map,x,y,0,1)
                        self.Astar_fill(m_map,x,y,1,1)
                        self.Astar_fill(m_map,x,y,1,0)
                    elif y == self.wid-1:
                        self.Astar_fill(m_map,x,y,-1,0)
                        self.Astar_fill(m_map,x,y,-1,-1)
                        self.Astar_fill(m_map,x,y,0,-1)
                        self.Astar_fill(m_map,x,y,1,0)
                        self.Astar_fill(m_map,x,y,1,-1)
                    else:
                        self.Astar_fill(m_map,x,y,-1,1)
                        self.Astar_fill(m_map,x,y,-1,0)
                        self.Astar_fill(m_map,x,y,-1,-1)
                        self.Astar_fill(m_map,x,y,0,1)
                        self.Astar_fill(m_map,x,y,0,-1)
                        self.Astar_fill(m_map,x,y,1,1)
                        self.Astar_fill(m_map,x,y,1,0)
                        self.Astar_fill(m_map,x,y,1,-1)
        return m_map

    def Astar_fill(self,m_map,x,y,dx,dy):
        if self.map[x+dx][y+dy] == 1:
            return None
        if dx == -1 and dy == 1:
            m_map[self.Astar_trans(x,y,2)][self.Astar_trans(x-1,y+1,3)] = 1
            m_map[self.Astar_trans(x,y,3)][self.Astar_trans(x-1,y+1,3)] = 1
            m_map[self.Astar_trans(x,y,4)][self.Astar_trans(x-1,y+1,3)] = 1
            m_map[self.Astar_trans(x,y,0)][self.Astar_trans(x-1,y+1,7)] = 1
            m_map[self.Astar_trans(x,y,7)][self.Astar_trans(x-1,y+1,7)] = 1
            m_map[self.Astar_trans(x,y,6)][self.Astar_trans(x-1,y+1,7)] = 1
        elif dx == 0 and dy == 1:
            m_map[self.Astar_trans(x,y,1)][self.Astar_trans(x,y+1,2)] = 1
            m_map[self.Astar_trans(x,y,2)][self.Astar_trans(x,y+1,2)] = 1
            m_map[self.Astar_trans(x,y,3)][self.Astar_trans(x,y+1,2)] = 1
            m_map[self.Astar_trans(x,y,5)][self.Astar_trans(x,y+1,6)] = 1
            m_map[self.Astar_trans(x,y,6)][self.Astar_trans(x,y+1,6)] = 1
            m_map[self.Astar_trans(x,y,7)][self.Astar_trans(x,y+1,6)] = 1
        elif dx == 1 and dy == 1:
            m_map[self.Astar_trans(x,y,0)][self.Astar_trans(x+1,y+1,1)] = 1
            m_map[self.Astar_trans(x,y,1)][self.Astar_trans(x+1,y+1,1)] = 1
            m_map[self.Astar_trans(x,y,2)][self.Astar_trans(x+1,y+1,1)] = 1
            m_map[self.Astar_trans(x,y,4)][self.Astar_trans(x+1,y+1,5)] = 1
            m_map[self.Astar_trans(x,y,5)][self.Astar_trans(x+1,y+1,5)] = 1
            m_map[self.Astar_trans(x,y,6)][self.Astar_trans(x+1,y+1,5)] = 1
        elif dx == -1 and dy == 0:
            m_map[self.Astar_trans(x,y,1)][self.Astar_trans(x-1,y,0)] = 1
            m_map[self.Astar_trans(x,y,0)][self.Astar_trans(x-1,y,0)] = 1
            m_map[self.Astar_trans(x,y,7)][self.Astar_trans(x-1,y,0)] = 1
            m_map[self.Astar_trans(x,y,3)][self.Astar_trans(x-1,y,4)] = 1
            m_map[self.Astar_trans(x,y,4)][self.Astar_trans(x-1,y,4)] = 1
            m_map[self.Astar_trans(x,y,5)][self.Astar_trans(x-1,y,4)] = 1
        elif dx == 1 and dy == 0:
            m_map[self.Astar_trans(x,y,1)][self.Astar_trans(x+1,y,0)] = 1
            m_map[self.Astar_trans(x,y,0)][self.Astar_trans(x+1,y,0)] = 1
            m_map[self.Astar_trans(x,y,7)][self.Astar_trans(x+1,y,0)] = 1
            m_map[self.Astar_trans(x,y,3)][self.Astar_trans(x+1,y,4)] = 1
            m_map[self.Astar_trans(x,y,4)][self.Astar_trans(x+1,y,4)] = 1
            m_map[self.Astar_trans(x,y,5)][self.Astar_trans(x+1,y,4)] = 1
        elif dx == -1 and dy == -1:
            m_map[self.Astar_trans(x,y,0)][self.Astar_trans(x-1,y-1,1)] = 1
            m_map[self.Astar_trans(x,y,1)][self.Astar_trans(x-1,y-1,1)] = 1
            m_map[self.Astar_trans(x,y,2)][self.Astar_trans(x-1,y-1,1)] = 1
            m_map[self.Astar_trans(x,y,4)][self.Astar_trans(x-1,y-1,5)] = 1
            m_map[self.Astar_trans(x,y,5)][self.Astar_trans(x-1,y-1,5)] = 1
            m_map[self.Astar_trans(x,y,6)][self.Astar_trans(x-1,y-1,5)] = 1
        elif dx == 0 and dy == -1:
            m_map[self.Astar_trans(x,y,1)][self.Astar_trans(x,y-1,2)] = 1
            m_map[self.Astar_trans(x,y,2)][self.Astar_trans(x,y-1,2)] = 1
            m_map[self.Astar_trans(x,y,3)][self.Astar_trans(x,y-1,2)] = 1
            m_map[self.Astar_trans(x,y,5)][self.Astar_trans(x,y-1,6)] = 1
            m_map[self.Astar_trans(x,y,6)][self.Astar_trans(x,y-1,6)] = 1
            m_map[self.Astar_trans(x,y,7)][self.Astar_trans(x,y-1,6)] = 1
        elif dx == 1 and dy == -1:
            m_map[self.Astar_trans(x,y,2)][self.Astar_trans(x+1,y-1,3)] = 1
            m_map[self.Astar_trans(x,y,3)][self.Astar_trans(x+1,y-1,3)] = 1
            m_map[self.Astar_trans(x,y,4)][self.Astar_trans(x+1,y-1,3)] = 1
            m_map[self.Astar_trans(x,y,0)][self.Astar_trans(x+1,y-1,7)] = 1
            m_map[self.Astar_trans(x,y,7)][self.Astar_trans(x+1,y-1,7)] = 1
            m_map[self.Astar_trans(x,y,6)][self.Astar_trans(x+1,y-1,7)] = 1

            


    
    def step(self, action):
        #按照action给每个agent更新位置
        obs = []
        direct = []
        reward = []
        done = []
        info = []
        for (a, agent) in zip(action,self.agents):#action操作0-6,前进：左打轮1，直行2，右打轮3， 后退：右打轮4，直行5， 左打轮6
            agent.act = a
            #print(a)
            if agent.movable:
                if agent.act[2-1]==1:
                #if agent.act == 1:
                    x_new = agent.pos.x + agent.vel*np.cos(np.pi*self.theta[agent.direct])
                    y_new = agent.pos.y + agent.vel*np.sin(np.pi*self.theta[agent.direct])
                    direct_new = agent.direct
                    #print("act 1")
                elif agent.act[5-1] == 1:#后退
                #elif agent.act == 4:
                    x_new = agent.pos.x - agent.vel*np.cos(np.pi*self.theta[agent.direct])
                    y_new = agent.pos.y - agent.vel*np.sin(np.pi*self.theta[agent.direct])
                    direct_new = agent.direct
                    #print("act 1")
                elif agent.act[1-1] == 1 or agent.act[6-1] == 1:#左打轮前进
                #elif agent.act == 0 or agent.act == 5:
                    #print(agent.direct)
                    x_ctr = agent.pos.x - agent.r*np.sin(np.pi*self.theta[agent.direct])
                    y_ctr = agent.pos.y + agent.r*np.cos(np.pi*self.theta[agent.direct])
                    dlt = agent.vel / agent.r
                    if agent.act[1-1] == 1:
                    #if agent.act == 0:
                        theta_new = (self.theta[agent.direct] + dlt/(np.pi)) % 2
                        direct_new = round(theta_new*4)%8
                        x_new = x_ctr + agent.r*np.sin(np.pi*theta_new)
                        y_new = y_ctr - agent.r*np.cos(np.pi*theta_new)
                        #print("act 1")
                    elif agent.act[6-1] == 1:
                    #elif agent.act == 5:
                        theta_new = (self.theta[agent.direct] - dlt/(np.pi)) % 2
                        direct_new = round(theta_new*4)%8
                        x_new = x_ctr + agent.r*np.sin(np.pi*theta_new)
                        y_new = y_ctr - agent.r*np.cos(np.pi*theta_new)
                elif agent.act[3-1] == 1 or agent.act[4-1] == 1:
                #elif agent.act == 2 or agent.act == 3:
                    x_ctr = agent.pos.x + agent.r*np.sin(np.pi*self.theta[agent.direct])
                    y_ctr = agent.pos.y - agent.r*np.cos(np.pi*self.theta[agent.direct])
                    dlt = agent.vel / agent.r
                    if agent.act[3-1] == 1:#右前进打轮
                    #if agent.act == 2:
                        theta_new = (self.theta[agent.direct] - dlt/(np.pi)) % 2
                        direct_new = round(theta_new*4)%8
                        x_new = x_ctr - agent.r*np.sin(np.pi*theta_new)
                        y_new = y_ctr + agent.r*np.cos(np.pi*theta_new)
                    elif agent.act[4-1] == 1:
                    #elif agent.act == 3:
                        theta_new = (self.theta[agent.direct] + dlt/(np.pi)) % 2
                        direct_new = round(theta_new*4)%8
                        x_new = x_ctr - agent.r*np.sin(np.pi*theta_new)
                        y_new = y_ctr + agent.r*np.cos(np.pi*theta_new)
                    #检查碰撞
                #print(a,"(",agent.pos.x, agent.pos.y,")", "(",x_new,y_new,")")
                x_new = int(round(x_new))
                y_new = int(round(y_new)) 
                #print("action:", a, "direct:",agent.direct,"pos:", agent.pos.x, agent.pos.y, "new", x_new, y_new)
                if self.o_map[x_new][y_new] == True or x_new > self.len-1 or x_new<0 or y_new>self.wid-1 or y_new<0:
                    #print(self.o_map)
                    #if agent.pos.x != x_new and agent.pos.y != y_new: 
                    agent.crash = True
                    #print("map_crash x_new:",x_new, "y_new:", y_new)
                    x_new = agent.pos.x
                    y_new = agent.pos.y
                    direct_new = agent.direct
                    #agent.movable = False
                for other in self.agents:
                    if other == agent:
                        continue
                    if other.pos.x == x_new and other.pos.y == y_new:
                        other.crash = True
                        agent.crash = True
                        #print("car_crash")
                        x_new = agent.pos.x
                        y_new = agent.pos.y
                        direct_new = agent.direct
                #print(a,"(",agent.pos.x, agent.pos.y,")", "(",x_new,y_new,")")


                self.map[agent.pos.x][agent.pos.y] = 0
                agent.s_buffer.x = agent.pos.x
                agent.s_buffer.y = agent.pos.y
                agent.pos.x = x_new
                agent.pos.y = y_new
                agent.direct = direct_new
                self.map[agent.pos.x][agent.pos.y] = 1
            if agent.pos.x == agent.goal.x and agent.pos.y == agent.goal.y:
                agent.reach = True
                #agent.movable = False
        for agent in self.agents:
            obs.append(self.observation(agent))
            #direct.append([np.cos(np.pi*self.theta[agent.direct]), np.sin(np.pi*self.theta[agent.direct])])
            reward.append(self.reward(agent))
            done.append(agent.reach)
            info.append([agent.pos.x,agent.pos.y,agent.crash,agent.reach])

        return np.array(obs), np.array(reward), done, info
    
    '''
    def step(self,action):
        obs = []
        reward = []
        done = []
        info = []
        for (a, agent) in zip(action,self.agents):#action操作0-6,前进：左打轮1，直行2，右打轮3， 后退：右打轮4，直行5， 左打轮6
            agent.act = a
            #print(a)
            if agent.movable:
                if agent.act[0] == 1: #0~6前后左右 不动 不动
                    x_new = agent.pos.x
                    y_new = agent.pos.y+1
                elif agent.act[1] == 1:
                    x_new = agent.pos.x
                    y_new = agent.pos.y-1
                elif agent.act[2] == 1:
                    x_new = agent.pos.x-1
                    y_new = agent.pos.y
                elif agent.act[3] == 1:
                    x_new = agent.pos.x+1
                    y_new = agent.pos.y
                else:
                    x_new = agent.pos.x
                    y_new = agent.pos.y
                if self.o_map[x_new][y_new] == True or x_new > self.len-1 or x_new<0 or y_new>self.wid-1 or y_new<0:
                    #print(self.o_map)
                    #if agent.pos.x != x_new and agent.pos.y != y_new: 
                    agent.crash = True
                    #print("map_crash x_new:",x_new, "y_new:", y_new)
                    x_new = agent.pos.x
                    y_new = agent.pos.y
                    direct_new = agent.direct
                    #agent.movable = False
                for other in self.agents:
                    if other == agent:
                        continue
                    if other.pos.x == x_new and other.pos.y == y_new:
                        other.crash = True
                        agent.crash = True
                        #print("car_crash")
                        x_new = agent.pos.x
                        y_new = agent.pos.y
                        direct_new = agent.direct
                #print(a,"(",agent.pos.x, agent.pos.y,")", "(",x_new,y_new,")")


                self.map[agent.pos.x][agent.pos.y] = 0
                agent.s_buffer.x = agent.pos.x
                agent.s_buffer.y = agent.pos.y
                agent.pos.x = x_new
                agent.pos.y = y_new
                #agent.direct = direct_new
                self.map[agent.pos.x][agent.pos.y] = 1
            if agent.pos.x == agent.goal.x and agent.pos.y == agent.goal.y:
                agent.reach = True
                #agent.movable = False
        for agent in self.agents:
            obs.append(self.observation(agent))
            reward.append(self.reward(agent))
            done.append(agent.reach)
            info.append([agent.pos.x,agent.pos.y,agent.crash,agent.reach])

        return np.array(obs), np.array(reward), done, info
    '''
                
    def reset(self):#怎么随机重置？
        start_list = []
        goal_list = []
        for agent in self.agents:
            agent.act = None
            agent.movable = True
            agent.reach = False
            agent.crash = False
            agent.vel = 1.47

            #agent.direct = 0
            #agent.pos.x = 1
            #agent.pos.y = 1
            #agent.goal.x = 2
            #agent.goal.y = 1
            #goal_list.append(Coordinate(2, 1))

            agent.direct = round(random.randint(0,7))
            agent.pos.x = round(random.randint(1,self.len-2))
            agent.pos.y = round(random.randint(1,self.wid-2))
            while([agent.pos.x, agent.pos.y] in start_list):
                agent.pos.x = round(random.randint(1,self.len-2))
                agent.pos.y = round(random.randint(1,self.wid-2))
            start_list.append([agent.pos.x, agent.pos.y])
            agent.goal.x = round(random.randint(1,self.len-2))
            agent.goal.y = round(random.randint(1,self.wid-2))
            while((agent.goal.x == agent.pos.x and agent.goal.y == agent.pos.y) or ([agent.goal.x, agent.pos.y] in goal_list)):
                agent.goal.x = round(random.randint(1,self.len-2))
                agent.goal.y = round(random.randint(1,self.wid-2))
            goal_list.append(Coordinate(agent.goal.x, agent.goal.y))
        self.goal_list = copy.deepcopy(goal_list)
        #空地图
        self.map = np.zeros((self.len, self.wid))
        self.map[0,:] = 1
        self.map[:,0] = 1
        self.map[self.len-1,:] = 1
        self.map[:,self.wid-1] = 1
        self.o_map = copy.deepcopy(self.map)
        

        obs = []
        #direct = []
        for agent in self.agents:
            obs.append(self.observation(agent))
            #direct.append([np.cos(np.pi*self.theta[agent.direct]), np.sin(np.pi*self.theta[agent.direct])])
        return np.array(obs)


    def reward(self,agent):#fai(x) - fai'(x)

        def dist(self, c1, c2):
            return math.sqrt((c1.x-c2.x)*(c1.x-c2.x) + (c1.y-c2.y)*(c1.y-c2.y))
        rew = 0
        #print(agent.pos.x, agent.pos.y, agent.s_buffer.x, agent.s_buffer.y)
        #self.rewcnt += 1
        #dists = math.sqrt((agent.pos.x-agent.goal.x)*(agent.pos.x-agent.goal.x)+(agent.pos.y-agent.goal.y)*(agent.pos.y-agent.goal.y))
        rew = -(dist(self, agent.pos, agent.goal) - dist(self, agent.s_buffer, agent.goal))
        if agent.crash:
            rew -= 5
            agent.crash = 0
            #print("erase crash")
        if agent.reach:
            rew += 5
            #self.rcnt += 1
            #print("agent reach")
        #if rew<-4:
        #    self.crashcnt += 1
        #    #print("crash_reward:", rew, "rewcnt:", self.rewcnt)
        #elif rew<0:
        #    self.mcnt += 1
        #else:
        #    self.pcnt +=1
        #
        #if self.rewcnt%100==0:
        #    print("crash:",self.crashcnt, "-reward:",self.mcnt,"reach:",self.rcnt, "+reward:",self.pcnt)
        #    self.crashcnt = 0
        #    self.mcnt = 0
        #    self.pcnt = 0
        #    self.rcnt = 0

        #print(rew)
        return rew
    
    def observation(self,agent):
        map = []#四通道图，自己的位置，自己的目标，障碍地图，其他agent的位置
        map_self_pos = np.zeros((self.len, self.wid))
        map_self_pos[agent.pos.x][agent.pos.y] = 1
        map.append(map_self_pos)
        map_self_goal = np.zeros((self.len, self.wid))
        map_self_goal[agent.goal.x][agent.goal.y] = 1
        map.append(map_self_goal)
        #障碍地图
        map.append(self.o_map)
        map_other_pos = np.zeros((self.len, self.wid))
    
        for other in self.agents:
            if other is agent: continue
            map_other_pos[other.pos.x][other.pos.y] = 1
        map.append(map_other_pos)
        direct1 = np.zeros((self.len,self.wid))
        direct2 = np.zeros((self.len,self.wid))
        direct1[:][:] = np.cos(np.pi*self.theta[agent.direct])
        direct2[:][:] = np.sin(np.pi*self.theta[agent.direct])
        map.append(direct1)
        map.append(direct2)
        #print("direct1:", direct1)
        #print("direct2:", direct2)
        #map = np.stack((map_self_pos, map_self_goal, self.o_map, map_other_pos),axis=2)
        return map
                
    def render(self, mode = 'human'):
        if self.viewer is None:
            from multiagent_particle_envs.multiagent import rendering
            self.viewer = rendering.Viewer(800,800)

        if self.agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from multiagent_particle_envs.multiagent import rendering
            self.viewer.set_bounds(0-self.cam_range, 0+self.cam_range, 0-self.cam_range, 0+self.cam_range)
            self.agent_geom_list = []
            self.grid_geom_list = []
            for agent in self.agents:
                agent_geom = {}
                total_xform = rendering.Transform()
                agent_geom['total_xform'] = total_xform

                half_l = agent.L_car/2.0
                half_w = agent.W_car/2.0
                geom = rendering.make_polygon([[half_l,half_w],[-half_l,half_w],[-half_l,-half_w],[half_l,-half_w]])
                geom.set_color(*agent.color,alpha = 0.4)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['car']=(geom,xform)

                #geom = rendering.make_line((0,0),(half_l,0))
                #geom.set_color(1.0,0.0,0.0,alpha = 1)
                #xform = rendering.Transform()
                #geom.add_attr(xform)
                #geom.add_attr(total_xform)
                #agent_geom['front_line']=(geom,xform)
                self.agent_geom_list.append(agent_geom)
            for x in range(self.wid):
                for y in range(self.len):
                    grid_geom={}
                    total_xform = rendering.Transform()
                    grid_geom['total_xform'] = total_xform

                    geom = rendering.make_polygon([[x+0.5,y+0.5],[x-0.5, y+0.5], [x-0.5, y-0.5], [x+0.5, y-0.5]])
                    if self.map[x][y] == 1:
                        geom.set_color(255,255,255, alpha = 0.4)
                    elif self.map[x][y] == 0:
                        geom.set_color(0,255,0,alpha=0.5)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    grid_geom['grid']=(geom,xform)
                    self.grid_geom_list.append(grid_geom)

            self.viewer.geoms = []
            for agent_geom in self.agent_geom_list:
                #self.viewer.add_geom(agent_geom['target_circle'][0])
                #for geom in agent_geom['laser_line']:
                #    self.viewer.add_geom(geom[0])
                self.viewer.add_geom(agent_geom['car'][0])
            for grid_geom in self.grid_geom_list:
                self.viewer.add_geom(grid_geom['grid'][0])
                #self.viewer.add_geom(agent_geom['front_line'][0])
                #self.viewer.add_geom(agent_geom['back_line'][0])
        for agent,agent_geom in zip(self.agents,self.agent_geom_list):
            
            #for idx,laser_line in enumerate(agent_geom['laser_line']):
            #        laser_line[1].set_scale(agent.laser_state[idx],agent.laser_state[idx]) 
            #agent_geom['front_line'][1].set_rotation(agent.state.phi)
            #agent_geom['target_circle'][1].set_translation(agent.state.target_x,agent.state.target_y)
            agent_geom['total_xform'].set_rotation(np.pi*self.theta[agent.direct])
            agent_geom['total_xform'].set_translation(agent.pos.x,agent.pos.y)
            
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        


                

            
        



            