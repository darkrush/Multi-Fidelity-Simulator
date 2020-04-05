from .basic import AgentState,Action,Observation,Fence
import numpy as np
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

def place_wall(cord,car_width,pixel_size):
    first_direction = 0 if cord[2]>cord[3] else 1    
    #first_direction = np.random.randint(2)
    placed_cord = -1
    if cord[2+first_direction] > 2*car_width:
        placed_cord = first_direction
    elif cord[3-first_direction] > 2*car_width:
        placed_cord = 1-first_direction
    if placed_cord > -1:
        
        wall_cord = np.clip(np.random.randn()/5.0,-1.0,1.0)*(cord[2+placed_cord]/2.0-car_width)+cord[2+placed_cord]/2.0
        wall = [placed_cord,wall_cord,pixel_size]
        first_cord = cord.copy()
        first_cord[2+placed_cord] = wall_cord
        second_cord = cord.copy()
        second_cord[placed_cord] = second_cord[placed_cord] + wall_cord
        second_cord[2+placed_cord] = second_cord[2+placed_cord] - wall_cord
        return [wall,first_cord,second_cord]
    else:
        return None

def random_map(map_W, map_H, pixel_size, car_width,door_width, wall_number, agent_number):
    placed_wall_number=0
    root_room = {'cord':[-map_W/2,-map_H/2,map_W,map_H], 'edge':[None,None,None,None], 'wall':None, 'door':None, 'children':None, 'parent':None}
    dead_count = 1000
    room_list = [root_room]
    while placed_wall_number<wall_number:
        placed = False
        for current_id in range(len(room_list)):
            if room_list[current_id]['wall'] is None:
                wall = place_wall(room_list[current_id]['cord'],car_width,pixel_size)
                if wall is not None:
                    placed = True
                    break

        if placed:
            first_children_edge = room_list[current_id]['edge'].copy()
            second_children_edge = room_list[current_id]['edge'].copy()
            if wall[0][0]==0:
                first_children_edge[1] = current_id
                second_children_edge[0] = current_id
            else:
                first_children_edge[3] = current_id
                second_children_edge[2] = current_id

            first_children = {'cord':wall[1], 'edge':first_children_edge, 'wall':None, 'door':None, 'children':None, 'parent':room_list[current_id]}
            second_children = {'cord':wall[2], 'edge':second_children_edge, 'wall':None, 'door':None, 'children':None, 'parent':room_list[current_id]}
            room_list[current_id]['wall'] = wall[0]
            room_list[current_id]['children'] = [first_children,second_children]

            room_list.append(first_children)
            room_list.append(second_children)

            placed_wall_number = placed_wall_number + 1
        else:
            raise Exception("No leaf_room can place wall")
        
        dead_count = dead_count-1
        if dead_count==0:
            raise Exception("failed")
    
    for current_id in range(len(room_list)):
        current_cord = room_list[current_id]['cord']
        current_wall = room_list[current_id]['wall']
        if current_wall is None:
            continue
        connect_wall_list = []
        for room_id in range(current_id+1,len(room_list)):
            room = room_list[room_id]
            if room['wall'] is None:
                continue
            if current_id in room['edge'] and not room['wall'][0]==current_wall[0]:
                connect_wall_list.append(room['wall'][1]+room['cord'][1-current_wall[0]])
        placed = False
        dead_count = 1000
        while not placed:
            if current_cord[current_wall[0]+2]<door_width+pixel_size:
                raise Exception("failed")
            rand_range = current_cord[1-current_wall[0]+2]-door_width-pixel_size
            start_cord = current_cord[1-current_wall[0]]
            door_cord = np.random.rand()*rand_range+pixel_size/2.0
            check_wall = True
            for wall_cord in connect_wall_list:
                right_beyound = wall_cord+pixel_size/2<door_cord+start_cord
                left_beyound =   wall_cord-pixel_size/2 >door_cord+door_width+start_cord
                if not (right_beyound or  left_beyound):
                    check_wall= False
                    break
            if check_wall:
                placed = True
            dead_count = dead_count -1
            if dead_count == 0:
                raise Exception("failed")
        room_list[current_id]['door'] =[door_cord,door_width]
            
    return room_list#root_room


def room2fence(room_list):
    basic_fence_prop = {}
    basic_fence_prop['color'] = [0, 0, 0]
    basic_fence_prop['anchor_x'] = 0
    basic_fence_prop['anchor_y'] = 0
    basic_fence_prop['rotation'] = 0
    basic_fence_prop['filled'] = True
    basic_fence_prop['close'] = True
    Fence_list = {}
    root_room = room_list[0]
    next_node = [root_room]
    wall_list = []
    while len(next_node)>0:
        if next_node[0]['wall'] is not None:
            wall_list.append((next_node[0]['cord'],next_node[0]['wall'],next_node[0]['door']))
        if next_node[0]['children'] is not None:
            next_node.append(next_node[0]['children'][0])
            next_node.append(next_node[0]['children'][1])
        next_node.pop(0)
    x=root_room['cord'][0]
    y=root_room['cord'][1]
    w=root_room['cord'][2]
    h=root_room['cord'][3]
    vertices = [[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]]
    edge_fence = basic_fence_prop.copy()
    edge_fence['vertices_x'] = [x for (x,y) in vertices]
    edge_fence['vertices_y'] = [y for (x,y) in vertices]
    edge_fence['color'] = [1, 1, 1]
    edge_fence['fill'] = False
    Fence_list[len(Fence_list)] = edge_fence
    for (cord,wall,door) in wall_list:
        x = cord[0]
        y = cord[1]
        w = cord[2]
        h = cord[3]
        wc = wall[1]
        wall_width = wall[2]
        if door is None:
            if wall[0] is 0:
                vertices = [[x+wc-wall_width,y],[x+wc+wall_width,y],[x+wc+wall_width,y+h],[x+wc-wall_width,y+h]]
            else:
                vertices = [[x,y+wc-wall_width],[x,y+wc+wall_width],[x+w,y+wc+wall_width],[x+w,y+wc-wall_width]]
            wall_fence = basic_fence_prop.copy()
            wall_fence['vertices_x'] = [x for (x,y) in vertices]
            wall_fence['vertices_y'] = [y for (x,y) in vertices]
            Fence_list[len(Fence_list)] = wall_fence
        else:
            dc = door[0]
            dw = door[1]
            if wall[0] is 0:
                #vertices = [[x+wc-wall_width,y+dc],[x+wc+wall_width,y+dc],[x+wc+wall_width,y+dc+dw],[x+wc-wall_width,y+dc+dw]]
                vertices1 = [[x+wc-wall_width,y],[x+wc+wall_width,y],[x+wc+wall_width,y+dc],[x+wc-wall_width,y+dc]]
                vertices2 = [[x+wc-wall_width,y+dc+dw],[x+wc+wall_width,y+dc+dw],[x+wc+wall_width,y+h],[x+wc-wall_width,y+h]]
            else:
                #vertices = [[x+dc,y+wc-wall_width],[x+dc,y+wc+wall_width],[x+dc+dw,y+wc+wall_width],[x+dc+dw,y+wc-wall_width]]
                vertices1 = [[x,y+wc-wall_width],[x,y+wc+wall_width],[x+dc,y+wc+wall_width],[x+dc,y+wc-wall_width]]
                vertices2 = [[x+dc+dw,y+wc-wall_width],[x+dc+dw,y+wc+wall_width],[x+w,y+wc+wall_width],[x+w,y+wc-wall_width]]
            wall_fence = basic_fence_prop.copy()
            wall_fence['vertices_x'] = [x for (x,y) in vertices1]
            wall_fence['vertices_y'] = [y for (x,y) in vertices1]
            Fence_list[len(Fence_list)] = wall_fence
            wall_fence = basic_fence_prop.copy()
            wall_fence['vertices_x'] = [x for (x,y) in vertices2]
            wall_fence['vertices_y'] = [y for (x,y) in vertices2]
            Fence_list[len(Fence_list)] = wall_fence
    return  Fence_list

def random_agent(room_list, agent_number, Agent_prop = None):
    if Agent_prop is None:
        Agent_prop = temp_agent_prop()
    
    leaf_room_list = [room for room in room_list if room['children'] is None]
    room_size_list = []
    R_Safe = Agent_prop['R_safe']
    for leaf_room in leaf_room_list:
        x,y,w,h = leaf_room['cord']
        wall_width = []
        for i in range(4):
            print(leaf_room['edge'][i])
            if leaf_room['edge'][i] is not None:
                print(room_list[leaf_room['edge'][i]]['wall'])
            wall_width.append(0 if leaf_room['edge'][i] is None else room_list[leaf_room['edge'][i]]['wall'][2])
        x = x+wall_width[0]+R_Safe
        y = y+wall_width[2]+R_Safe
        w = w - wall_width[0] - wall_width[1]-2*R_Safe
        h = h - wall_width[2] - wall_width[3]-2*R_Safe
        room_size_list.append([x,y,w,h])
    dead_count = 1000
    agent_cord_list = {'agent_init_cord':[],'agent_target_cord':[]}
    for cord_list_id in ['agent_init_cord','agent_target_cord']:
        while dead_count>0:
            room_id_list = [np.random.randint(len(room_size_list)) for _ in range(agent_number)]
            x_list = [room_size_list[room_id][0]+ room_size_list[room_id][2]*np.random.rand() for room_id in room_id_list]
            y_list = [room_size_list[room_id][1]+ room_size_list[room_id][3]*np.random.rand() for room_id in room_id_list]
            failed = False
            for pos_id_1 in range(agent_number):
                for pos_id_2 in range(pos_id_1+1,agent_number):
                    dist_squre = (x_list[pos_id_1]-x_list[pos_id_2])**2+(y_list[pos_id_1]-y_list[pos_id_2])**2
                    if dist_squre<(2*R_Safe)**2:
                        failed = True
                        break
                if failed :
                    break
            if not failed:
                break
            dead_count = dead_count - 1
            if dead_count == 0:
                raise Exception("failed")
        agent_cord_list[cord_list_id] = [[x,y]for (x,y) in zip(x_list,y_list)]
    main_group = []
    for agent_id in range(agent_number):
        agent_prop = Agent_prop.copy()
        agent_prop['init_x'] = agent_cord_list['agent_init_cord'][agent_id][0]
        agent_prop['init_y'] = agent_cord_list['agent_init_cord'][agent_id][1]
        agent_prop['init_theta'] = np.random.rand()*6.28
        agent_prop['init_vel_b'] = 0
        agent_prop['init_phi'] = 0


        agent_prop['init_target_x'] = agent_cord_list['agent_target_cord'][agent_id][0]
        agent_prop['init_target_y'] = agent_cord_list['agent_target_cord'][agent_id][1]
        main_group.append(agent_prop)

    Agent_list = {'main_group':main_group}
    return Agent_list

def render(room_list):
    from . import rendering 
    viewer = rendering.Viewer(int(400),int(400))
    viewer.set_bounds(-5,+5,-5,+5)
    root_room = room_list[0]
    next_node = [root_room]
    wall_list = []
    while len(next_node)>0:
        if next_node[0]['wall'] is not None:
            wall_list.append((next_node[0]['cord'],next_node[0]['wall'],next_node[0]['door']))
        if next_node[0]['children'] is not None:
            next_node.append(next_node[0]['children'][0])
            next_node.append(next_node[0]['children'][1])
        next_node.pop(0)
    x=root_room['cord'][0]
    y=root_room['cord'][1]
    w=root_room['cord'][2]
    h=root_room['cord'][3]
    vertices = [[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]]
    geom = rendering.make_polyline(vertices)
    viewer.add_geom(geom)
    for (cord,wall,door) in wall_list:
        x = cord[0]
        y = cord[1]
        w = cord[2]
        h = cord[3]
        wc = wall[1]
        wall_width = 0.5/2
        if door is None:
            if wall[0] is 0:
                vertices = [[x+wc-wall_width,y],[x+wc+wall_width,y],[x+wc+wall_width,y+h],[x+wc-wall_width,y+h]]
            else:
                vertices = [[x,y+wc-wall_width],[x,y+wc+wall_width],[x+w,y+wc+wall_width],[x+w,y+wc-wall_width]]
            geom = rendering.make_polygon(vertices)
            viewer.add_geom(geom)
        else:
            dc = door[0]
            dw = door[1]
            if wall[0] is 0:
                vertices1 = [[x+wc-wall_width,y],[x+wc+wall_width,y],[x+wc+wall_width,y+dc],[x+wc-wall_width,y+dc]]
                vertices2 = [[x+wc-wall_width,y+dc+dw],[x+wc+wall_width,y+dc+dw],[x+wc+wall_width,y+h],[x+wc-wall_width,y+h]]
            else:
                vertices1 = [[x,y+wc-wall_width],[x,y+wc+wall_width],[x+dc,y+wc+wall_width],[x+dc,y+wc-wall_width]]
                vertices2 = [[x+dc+dw,y+wc-wall_width],[x+dc+dw,y+wc+wall_width],[x+w,y+wc+wall_width],[x+w,y+wc-wall_width]]
            geom1 = rendering.make_polygon(vertices1)
            geom2 = rendering.make_polygon(vertices2)
            viewer.add_geom(geom1)
            viewer.add_geom(geom2)
            
    viewer.render(return_rgb_array = False)
#if __name__=='__main__':
#    print(place_wall())