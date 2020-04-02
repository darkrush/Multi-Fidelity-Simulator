import numpy as np
class AgentState(object):
    def __init__(self):
        #center point position in x,y axis
        self.x = 0
        self.y = 0
        #linear velocity of back point
        self.vel_b = 0
        # direction of car axis
        self.theta = 0
        # Deflection angle of front wheel
        self.phi = 0
        self.enable = True
        # Movable
        self.movable = True
        self.crash = False
        self.reach = False
        # target x coordinate
        self.target_x   = 1
        # target y coordinate
        self.target_y   = 1


# action of the agent
class Action(object):
    def __init__(self):
        self.ctrl_vel = 0 # ctrl_vel \belong [-1,1]
        self.ctrl_phi = 0 # ctrl_phi \belong [-1,1]
    
    def __str__(self):
        return 'ctrl_vel : '+str(self.ctrl_vel)+' ctrl_phi : '+str(self.ctrl_phi)
    def __repr__(self):
        return self.__str__()

class Observation(object):
    def __init__(self):
        self.pos = [0,0,0,0,0] # x,y,theta,target_x,target_y
        self.laser_data = []   # float*n

    def __str__(self):
        return ' pos : '+str(self.pos)+' laser : '+str(self.laser_data)
    def __repr__(self):
        return self.__str__()

# properties of agent entities
class AgentProp(object):
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

class Fence(object):
    def __init__(self,fence_prop):# anchor=[0,0], rotation = 0, vertices=([0,0],), close = False, filled = False, color = [0.0, 0.0, 0.0]):
        
        # the anchor point in global coordinate
        self.anchor = [fence_prop['anchor_x'],fence_prop['anchor_y']]
        # the rotation angle by radian global coordinate
        self.rotation = fence_prop['rotation']
        # the coordinate of vertices related to anchor
        self.vertices = zip(fence_prop['vertices_x'],fence_prop['vertices_y'])
        # Fill the fence by color inside if True
        self.filled = fence_prop['filled']
        # A close fence means a fence between vertices[-1] and vertices[0], forced to be True if filled 
        self.close = fence_prop['close'] or fence_prop['filled']
        # color
        self.color = fence_prop['color']

        self.calc_vertices()

    def calc_vertices(self):
        self.global_vertices = []
        for v in self.vertices:
            c = np.cos(self.rotation)
            s = np.sin(self.rotation)
            g_v_x = v[0]*c - v[1]*s +self.anchor[0]
            g_v_y = v[1]*c + v[0]*s +self.anchor[1]
            self.global_vertices.append(np.array([g_v_x,g_v_y]))
        if self.close:
            self.global_vertices.append(self.global_vertices[0])