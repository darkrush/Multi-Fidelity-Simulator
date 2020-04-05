import numpy as np
import math
from .basic import AgentState,Action,Observation,Fence

# properties of agent entities
class Agent(object):
    def __init__(self,agent_prop = None):
        self.R_safe     = 0.2  # minimal distance not crash
        self.R_reach    = 0.1  # maximal distance for reach target
        self.L_car      = 0.3  # length of the car
        self.W_car      = 0.2  # width of the car
        self.L_axis     = 0.25 # distance between front and back wheel
        self.R_laser    = 4    # range of laser
        self.N_laser    = 360  # number of laser lines
        self.K_vel      = 1    # coefficient of back whell velocity control
        self.K_phi      = 30   # coefficient of front wheel deflection control
        self.init_x     = -1   # init x coordinate
        self.init_y     = -1   # init y coordinate
        self.init_theta = 0    # init theta
        self.init_vel_b = 0    # init velocity of back point
        self.init_phi   = 0    # init front wheel deflection
        self.init_movable = True # init movable state
        self.init_target_x = 1 # init target x coordinate
        self.init_target_y = 1 # init target y coordinate

        if agent_prop is not None:
            for k,v in agent_prop.items():
                self.__dict__[k] = v
        self.N_laser = int(self.N_laser)

        self.state = AgentState()
        self.action = Action()
        self.color = [0,0,0]
        #temp laser state
        self.laser_state = np.array([self.R_laser]*self.N_laser)

        self.reset()

    def reset(self,state = None):
        self.total_time = 0
        if state is not None:
            self.state = state
        else:
            self.state.x = self.init_x
            self.state.y = self.init_y
            self.state.theta = self.init_theta
            self.state.vel_b = self.init_vel_b
            self.state.phi   = self.init_phi
            self.state.movable     = self.init_movable
            self.state.crash = False
            self.state.reach = False
        self.laser_state = np.array([self.R_laser]*self.N_laser)
        


    def check_AA_collisions(self,agent_b):
        min_dist = (self.R_safe + agent_b.R_safe)**2
        ab_dist = (self.state.x - agent_b.state.x)**2 + (self.state.y - agent_b.state.y)**2
        return ab_dist<=min_dist
    
    def check_AF_collisions(self,fence):
        r = self.R_safe
        o_pos =  [self.state.x,self.state.y]
        for i in range(len(fence.global_vertices)-1):
            a_pos = fence.global_vertices[i]
            v_oa = a_pos - o_pos
            av_dist = np.linalg.norm(v_oa)
            #crash if a vertex inside agent
            if av_dist<=r:
                return True
            b_pos = fence.global_vertices[i+1]
            v_ab = b_pos - a_pos
            v_ob = b_pos - o_pos
            #crash  if two vertex in different sides of perpendicular and d(o,ab)<r
            if (np.dot(v_oa,v_ab)<0) and (np.dot(v_ob,-v_ab)<0):
                dist_o_ab = np.abs(np.cross(v_oa,v_ab)/np.linalg.norm(v_ab))
                if dist_o_ab <= r:
                    return True
        return False

    def check_reach(self):
        max_dist = self.R_reach**2
        at_dist = (self.state.x - self.state.target_x)**2 + (self.state.y - self.state.target_y)**2
        return at_dist<=max_dist
        
    def laser_agent_agent(self,agent_b):
        R = self.R_laser
        N = self.N_laser
        l_laser = np.array([R]*N)
        o_pos =  np.array([self.state.x,self.state.y])
        oi_pos = np.array([agent_b.state.x,agent_b.state.y])
        if np.linalg.norm(o_pos-oi_pos)>R+(agent_b.L_car**2 + agent_b.W_car**2)**0.5 / 2.0:
            return l_laser
        theta = self.state.theta
        theta_b = agent_b.state.theta
        cthb= math.cos(theta_b)
        sthb= math.sin(theta_b)
        half_l_shift = np.array([cthb,sthb])*agent_b.L_car/2.0
        half_w_shift = np.array([-sthb,cthb])*agent_b.W_car/2.0
        car_points = []
        car_points.append(oi_pos+half_l_shift+half_w_shift-o_pos)
        car_points.append(oi_pos-half_l_shift+half_w_shift-o_pos)
        car_points.append(oi_pos-half_l_shift-half_w_shift-o_pos)
        car_points.append(oi_pos+half_l_shift-half_w_shift-o_pos)
        car_line = [[car_points[i],car_points[(i+1)%len(car_points)]] for i in range(len(car_points))]
        for start_point, end_point in  car_line:
            v_es = start_point-end_point
            tao_es = np.array((v_es[1],-v_es[0]))
            tao_es = tao_es/np.linalg.norm(tao_es)
            if abs(np.dot(start_point,tao_es))>R:
                continue
            if np.cross(start_point,end_point) < 0 :
                start_point,end_point = end_point,start_point
            theta_start = np.arccos(start_point[0]/np.linalg.norm(start_point))
            if start_point[1]<0:
                theta_start = math.pi*2-theta_start
            theta_start-=theta
            theta_end = np.arccos(end_point[0]/np.linalg.norm(end_point))
            if end_point[1]<0:
                theta_end = math.pi*2-theta_end
            theta_end-=theta
            laser_idx_start = theta_start/(2*math.pi/N)
            laser_idx_end   =   theta_end/(2*math.pi/N)
            if laser_idx_start> laser_idx_end:
                laser_idx_end+=N
            if math.floor(laser_idx_end)-math.floor(laser_idx_start)==0:
                continue
            laser_idx_start = math.ceil(laser_idx_start)
            laser_idx_end = math.floor(laser_idx_end)
            for laser_idx in range(laser_idx_start,laser_idx_end+1):
                laser_idx%=N
                x1 = start_point[0]
                y1 = start_point[1]
                x2 = end_point[0]
                y2 = end_point[1]
                theta_i = theta+laser_idx*math.pi*2/N
                cthi = math.cos(theta_i)
                sthi = math.sin(theta_i)
                temp = (y1-y2)*cthi - (x1-x2)*sthi
                # temp equal zero when collinear
                if abs(temp) <= 1e-10:
                    dist = R 
                else:
                    dist = (x2*y1-x1*y2)/(temp)
                if dist > 0:
                    l_laser[laser_idx] = min(l_laser[laser_idx],dist)
        return l_laser

    def laser_agent_fence(self,fence):
        R = self.R_laser
        N = self.N_laser
        l_laser = np.array([R]*N)
        o_pos =  np.array([self.state.x,self.state.y])
        theta = self.state.theta
        for i in range(len(fence.global_vertices)-1):
            a_pos = fence.global_vertices[i]
            b_pos = fence.global_vertices[i+1]
            oxaddR = o_pos[0]+R
            oxsubR = o_pos[0]-R
            oyaddR = o_pos[1]+R
            oysubR = o_pos[1]-R
            if oxaddR<a_pos[0] and  oxaddR<b_pos[0]:
                continue
            if oxsubR>a_pos[0] and  oxsubR>b_pos[0]:
                continue
            if oyaddR<a_pos[1] and  oyaddR<b_pos[1]:
                continue
            if oysubR>a_pos[1] and  oysubR>b_pos[1]:
                continue
    
            v_oa = a_pos - o_pos
            v_ob = b_pos - o_pos
            v_ab = b_pos - a_pos
            print(np.linalg.norm(v_ab))
            print(np.cross(v_oa,v_ab))
            dist_o_ab = np.abs(np.cross(v_oa,v_ab)/np.linalg.norm(v_ab))

            #if distance(o,ab) > R, laser signal changes
            if dist_o_ab > R:
                continue
            S1 = np.cross(v_ab,-v_oa)
            aa = np.dot(v_oa,v_oa)
            bb = np.dot(v_ob,v_ob)
            ab = np.dot(v_oa,v_ob)
            numerator = ab*ab-aa*bb
            #v_ab_normal = np.array([v_ab[1],-v_ab[0]])
            #if np.dot(v_oa,v_ab_normal) < 0:
            #    v_ab_normal = -v_ab_normal
            max_adot = 0
            max_aid = 0
            max_bdot = 0
            max_bid = 0
            for idx_laser in range(N):
                theta_i = theta+idx_laser*math.pi*2/N
                c_x = R*np.cos(theta_i)
                c_y = R*np.sin(theta_i)
                v_oc = np.array([c_x,c_y])
                adot = np.dot(v_oc,v_oa)
                bdot = np.dot(v_oc,v_ob)
                if adot > max_adot:
                    max_adot = adot
                    max_aid = idx_laser
                if bdot > max_bdot:
                    max_bdot = bdot
                    max_bid = idx_laser
            if max_aid > max_bid:
                max_bid,max_aid = (max_aid,max_bid)
            if max_bid - max_aid > N//2:
                max_bid,max_aid = (max_aid,max_bid)
                max_bid+=N
                
            #range1 = N//2
            #range2 = N - range1
            for idx_laser in range(max_aid,max_bid+1):
                idx_laser %= N
                theta_i = theta+ idx_laser*math.pi*2/N
                c_x = R*np.cos(theta_i)
                c_y = R*np.sin(theta_i)            
                c_pos = np.array([c_x,c_y])+o_pos
                v_ac = c_pos - a_pos
                S2 = np.cross(v_ab,v_ac)
                if S1*S2>0:
                    continue
                v_oc = c_pos - o_pos
                if np.cross(v_oc,v_oa)*np.cross(v_oc,v_ob) >0:
                    continue
                cb = np.dot(v_oc,v_ob)
                ca = np.dot(v_oc,v_oa)
                denominator = (ab-bb)*ca-(aa-ab)*cb
                d = abs(numerator/denominator*np.linalg.norm(v_oc))
                l_laser[idx_laser] = min(l_laser[idx_laser],d)
        return l_laser
    
# multi-agent world
class World(object):
    def __init__(self,agent_groups,fence_list,cfg = None):
        
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        for (_,agent_group) in agent_groups.items():
            for agent_prop in agent_group:
                agent = Agent(agent_prop)
                self.agents.append(agent)
        self.fences = []
        for (_,fence_prop) in fence_list.items():
             fence = Fence(fence_prop)
             self.fences.append(fence)
        #for idx,agent in enumerate(self.agents):
            #agent.color = hsv2rgb(360.0/len(self.agents)*idx,1.0,1.0)
        # simulation timestep
        self.dt = 0.1
        self.window_scale = 1.0
        if cfg is not None:
            self.dt = cfg['dt']
            self.window_scale = cfg['window_scale']
        self.cam_range = 4
        self.viewer = None
        self.total_time = 0
        self._reset_render()
    def reset(self):
        self.total_time = 0
        for agent in self.agents:
            for k in agent.state.__dict__.keys():
                if k == 'reach' or k == 'crash':
                    continue
                agent.state.__dict__[k] = agent.__dict__['init_'+k]
            agent.state.crash = False
            agent.state.reach = False
        self._reset_render()
        return True

    def get_total_time(self):
        return self.total_time

    def get_agent_number(self):
        return len(self.agents)

    def set_action(self,enable_list,actions):
        for enable,agent,action in zip(enable_list,self.agents,actions):
            if not enable: continue
            agent.action.ctrl_vel = action.ctrl_vel
            agent.action.ctrl_phi = action.ctrl_phi

    def get_state(self):
        return [agent.state for agent in self.agents]
    
    def set_state(self,enable_list,states):
        for enable,agent,state in zip(enable_list,self.agents,states):
            if not enable :continue
            for k in agent.state.__dict__.keys():
                agent.state.__dict__[k] = state.__dict__[k]
    
    def get_obs(self):
        obs_data = {'time':self.total_time,'obs_data':[]}
        self.update_laser_state()
        for idx_a in range(len(self.agents)):
            state = self.agents[idx_a].state
            pos = np.array([state.x, state.y, state.theta, state.target_x, state.target_y])
            laser_data = self.agents[idx_a].laser_state
            obs = Observation()
            obs.pos = pos
            obs.laser_data = laser_data
            obs_data['obs_data'].append(obs)
        return obs_data['obs_data']

    # update state of the world
    def step(self):
        self.apply_action()
        self.integrate_state()
        self.check_collisions()
        self.check_reach()
        self.total_time += self.dt
    
    # gather agent action forces
    def apply_action(self):
        # set applied forces
        for agent in self.agents:
            agent.state.vel_b = np.clip(agent.action.ctrl_vel, -1.0, 1.0)*agent.K_vel if agent.state.movable else 0
            agent.state.phi   = np.clip(agent.action.ctrl_phi, -1.0, 1.0)*agent.K_phi if agent.state.movable else 0
    
    def update_laser_state(self):
        for idx_a,agent_a in enumerate(self.agents):
            agent_a.laser_state = np.array([agent_a.R_laser]*agent_a.N_laser)
            for fence in self.fences:
                l_laser = agent_a.laser_agent_fence(fence)
                agent_a.laser_state = np.min(np.vstack([agent_a.laser_state,l_laser]),axis = 0)
            for idx_b,agent_b in enumerate(self.agents):
                if idx_a == idx_b:
                    continue
                l_laser = agent_a.laser_agent_agent(agent_b)
                agent_a.laser_state = np.min(np.vstack([agent_a.laser_state,l_laser]),axis = 0)

    # integrate physical state
    def integrate_state(self):
        for agent in self.agents:
            if not agent.state.movable: continue
            _phi = agent.state.phi
            _vb = agent.state.vel_b
            _theta = agent.state.theta
            sth = math.sin(_theta)
            cth = math.cos(_theta)
            _L = agent.L_axis
            _xb = agent.state.x - cth*_L/2.0
            _yb = agent.state.y - sth*_L/2.0
            tphi = math.tan(_phi)
            _omega = _vb/_L*tphi
            _delta_theta = _omega * self.dt
            if abs(_phi)>0.00001:
                _rb = _L/tphi
                _delta_tao = _rb*(1-math.cos(_delta_theta))
                _delta_yeta = _rb*math.sin(_delta_theta)
            else:
                _delta_tao = _vb*self.dt*(_delta_theta/2.0)
                _delta_yeta = _vb*self.dt*(1-_delta_theta**2/6.0)
            
            _xb += _delta_yeta*cth - _delta_tao*sth
            _yb += _delta_yeta*sth + _delta_tao*cth
            _theta += _delta_theta
            _theta = (_theta/math.pi)%2*math.pi

            agent.state.x = _xb + math.cos(_theta)*_L/2.0
            agent.state.y = _yb + math.sin(_theta)*_L/2.0
            agent.state.theta = _theta
        
    def check_collisions(self):
        for ia, agent_a in enumerate(self.agents):
            if agent_a.state.crash :
                continue
            
            for fence in self.fences:
                if agent_a.check_AF_collisions(fence):
                    agent_a.state.crash = True
                    agent_a.state.movable = False
                    break

            for ib, agent_b in enumerate(self.agents):
                if ia==ib :
                    continue
                if agent_a.check_AA_collisions(agent_b) :
                    agent_a.state.crash = True
                    agent_a.state.movable = False
                    break
    
    def check_reach(self):
        for agent in self.agents:
            reach = agent.check_reach()
            if reach :
                agent.state.reach = True
                agent.state.movable = False
    
    def _reset_render(self):
        self.agent_geom_list = None
        self.fence_geom_list = None

    # render environment
    def render(self,time , mode='human'):
        if self.viewer is None:
            from . import rendering 
            self.viewer = rendering.Viewer(int(800*self.window_scale),int(800*self.window_scale))
        # create rendering geometry
        if self.agent_geom_list is None:
            # import rendering only if we need it (and don't import for headless machines)
            from . import rendering
            self.viewer.set_bounds(0-self.cam_range*self.window_scale, 0+self.cam_range*self.window_scale, 0-self.cam_range*self.window_scale, 0+self.cam_range*self.window_scale)
            self.agent_geom_list = []
            
            for agent in self.agents:
                agent_geom = {}
                total_xform = rendering.Transform()
                agent_geom['total_xform'] = total_xform
                agent_geom['laser_line'] = []

                geom = rendering.make_circle(agent.R_reach)
                geom.set_color(*agent.color)
                xform = rendering.Transform()
                geom.add_attr(xform)
                agent_geom['target_circle']=(geom,xform)

                N = agent.N_laser
                for idx_laser in range(N):
                    theta_i = idx_laser*math.pi*2/N
                    #d = agent.R_laser
                    d = 1
                    end = (math.cos(theta_i)*d, math.sin(theta_i)*d)
                    geom = rendering.make_line((0, 0),end)
                    geom.set_color(0.0,1.0,0.0,alpha = 0.5)
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.add_attr(total_xform)
                    agent_geom['laser_line'].append((geom,xform))
                
                half_l = agent.L_car/2.0
                half_w = agent.W_car/2.0
                geom = rendering.make_polygon([[half_l,half_w],[-half_l,half_w],[-half_l,-half_w],[half_l,-half_w]])
                geom.set_color(*agent.color,alpha = 0.4)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['car']=(geom,xform)

                geom = rendering.make_line((0,0),(half_l,0))
                geom.set_color(1.0,0.0,0.0,alpha = 1)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['front_line']=(geom,xform)
                
                geom = rendering.make_line((0,0),(-half_l,0))
                geom.set_color(0.0,0.0,0.0,alpha = 1)
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.add_attr(total_xform)
                agent_geom['back_line']=(geom,xform)

                self.agent_geom_list.append(agent_geom)

            self.fence_geom_list = []
            for fence in self.fences:
                if fence.filled :
                    geom = rendering.make_polygon(fence.global_vertices)
                    geom.set_color(*fence.color)
                else:
                    geom = rendering.make_polyline(fence.global_vertices)
                xform = rendering.Transform() 
                #geom.set_color(*fence.color)
                geom.add_attr(xform)
                self.fence_geom_list.append([geom,xform])
        
            #self.render_geoms_xform.append(xform)
            self.viewer.geoms = []
            for fence_geom in self.fence_geom_list:
                self.viewer.add_geom(fence_geom[0])
            for agent_geom in self.agent_geom_list:
                self.viewer.add_geom(agent_geom['target_circle'][0])
                for geom in agent_geom['laser_line']:
                    self.viewer.add_geom(geom[0])
                self.viewer.add_geom(agent_geom['car'][0])
                self.viewer.add_geom(agent_geom['front_line'][0])
                self.viewer.add_geom(agent_geom['back_line'][0])

            
                

        self.update_laser_state()
        for agent,agent_geom in zip(self.agents,self.agent_geom_list):
            
            for idx,laser_line in enumerate(agent_geom['laser_line']):
                    laser_line[1].set_scale(agent.laser_state[idx],agent.laser_state[idx]) 
            #agent_geom['front_line'][1].set_rotation(agent.state.phi)
            #agent_geom['target_circle'][1].set_translation(agent.state.target_x,agent.state.target_y)
            #agent_geom['total_xform'].set_rotation(agent.state.theta)
            #agent_geom['total_xform'].set_translation(agent.state.x,agent.state.y)
            
            agent_geom['front_line'][1].set_rotation(agent.state.phi)
            agent_geom['target_circle'][1].set_translation(agent.state.target_x*self.window_scale,agent.state.target_y*self.window_scale)
            agent_geom['target_circle'][1].set_scale(self.window_scale,self.window_scale)
            agent_geom['total_xform'].set_scale(self.window_scale,self.window_scale)
            agent_geom['total_xform'].set_rotation(agent.state.theta)
            agent_geom['total_xform'].set_translation(agent.state.x*self.window_scale,agent.state.y*self.window_scale)
        
        for fence_geom in self.fence_geom_list:
            fence_geom[1].set_scale(self.window_scale,self.window_scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')