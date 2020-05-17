#from .basic import AgentState,Action,Observation,Fence
import copy

def clip(x,l,u):
    
    if x < l:
        return l
    elif x > u:
        return u
    else:
        return x

def place_wall(coord,car_R,half_wall_width,np_random):
    # select long direction to place split wall
    place_direction = 0 if coord[2]>coord[3] else 1
    # check the place direction have enough distance
    if coord[2+place_direction] < (2*half_wall_width + 2*car_R)*2:
        return None
    # place wall and split the room into two    
    mid_coord = coord[2+place_direction]/2.0
    random_range = coord[2+place_direction]/2.0 - 2*car_R -2*half_wall_width
    wall_coord = clip(np_random.randn()/5.0,-1.0,1.0)*random_range + mid_coord
    # a wall is a list with [x,y,direction,length,half_wall_width]
    wall = {'coord':[coord[0],coord[1]],
            'direction':place_direction,
            'length':coord[3-place_direction],
            'hww':half_wall_width,
            #'door':[coord[3-place_direction]/3.0,coord[3-place_direction]/3.0]}
            'door':[]}
    wall['coord'][place_direction] = wall['coord'][place_direction] + wall_coord
    # calculate the coordinate for two childen 
    first_coord = coord.copy()
    first_coord[2+place_direction] = wall_coord
    second_coord = coord.copy()
    second_coord[place_direction] = second_coord[place_direction] + wall_coord
    second_coord[2+place_direction] = second_coord[2+place_direction] - wall_coord
    return [wall,first_coord,second_coord]



def random_room(map_W,map_H,half_wall_width, car_R, door_width, room_number,np_random, max_dead_count = 1000):
    assert door_width >= 2*car_R
    # a wall is a list with [x,y,direction,length,half_wall_width]
    
    wall_list = []
    wall = {'coord':[0,0],
            'direction':0,
            'length':map_H,
            'hww':half_wall_width,
            'door':[]}

    wall_list.append(wall.copy())#left
    wall['coord'] = [map_W,0]
    wall_list.append(wall.copy())#right
    wall['coord'] = [0,0]
    wall['direction'] = 1
    wall['length'] = map_W
    wall_list.append(wall.copy())#bottom
    wall['coord'] = [0,map_H]
    wall_list.append(wall.copy())#top

    room_list = []
    room_list.append({'coord':[0,0,map_W,map_H],'edge':[0,1,2,3]})
    while len(room_list)<room_number:
        max_length_id=[0,0]
        max_length = room_list[0]['coord'][2]
        for idx,room in enumerate(room_list):
            if room['coord'][2]>max_length:
                max_length_id = [idx,0]
                max_length = room['coord'][2]
            if room['coord'][3]>max_length:
                max_length_id = [idx,1]
                max_length = room['coord'][3]
        wall_result = place_wall(room_list[max_length_id[0]]['coord'],car_R,half_wall_width,np_random)
        if wall_result is None:
            print('cannot place enough room')
            break
        wall,fst_coord,scd_coord = wall_result
        wall_idx = len(wall_list)
        wall_list.append(wall)
        fst_edge = room_list[max_length_id[0]]['edge'].copy()
        scd_edge = room_list[max_length_id[0]]['edge'].copy()
        if wall['direction']==0:
            fst_edge[1] = wall_idx
            scd_edge[0] = wall_idx
        else:
            fst_edge[3] = wall_idx
            scd_edge[2] = wall_idx

        room_list.pop(max_length_id[0])
        room_list.append({'coord':fst_coord,'edge':fst_edge})
        room_list.append({'coord':scd_coord,'edge':scd_edge})
    
    placeable_area_list = []
    for room in room_list:
        x = room['coord'][0] + wall_list[room['edge'][0]]['hww']
        y = room['coord'][1] + wall_list[room['edge'][2]]['hww']
        w = room['coord'][2] - wall_list[room['edge'][0]]['hww'] - wall_list[room['edge'][1]]['hww']
        h = room['coord'][3] - wall_list[room['edge'][2]]['hww'] - wall_list[room['edge'][3]]['hww']
        placeable_area_list.append([x,y,w,h])

    return room_list, wall_list, placeable_area_list

def check_adjacency(room_1,room_2,door_width,half_wall_width,share_pair):
    direction = 1 if (share_pair[0] == 0 or share_pair[0] == 1) else 0
    low_cord = max(room_1['coord'][direction],room_2['coord'][direction])
    high_cord = min(room_1['coord'][direction] + room_1['coord'][direction+2], \
                    room_2['coord'][direction] + room_2['coord'][direction+2])
    if high_cord<low_cord+door_width+half_wall_width*2:
        return None
    return low_cord,high_cord

def place_door_wall(wall,low,high,half_wall_width, door_width,np_random):
    direction = wall['direction']
    start = wall['coord'][1-direction]
    random_range = high-low-2*half_wall_width-door_width
    random_start = low + half_wall_width
    coord = np_random.rand()*random_range + random_start - start
    return [coord,door_width]

def place_door(room_list, wall_list,half_wall_width, door_width,np_random, extra_door_prop = 0.5):
    share_idx_pair = [[0,1],[1,0],[2,3],[3,2]]
    adjacency_tabel = []
    #判定所有房间之间的有效邻接性（至少可以通过一扇门打通）
    for idx_1 in range(len(room_list)):
        for idx_2 in range(idx_1+1,len(room_list)):
            share = -1
            for share_idx in range(4):
                edge_1,edge_2 = share_idx_pair[share_idx]
                if room_list[idx_1]['edge'][edge_1] == room_list[idx_2]['edge'][edge_2]:
                    share = share_idx
                    wall_idx = room_list[idx_1]['edge'][edge_1]
                    break
            if share == -1:
                continue
            adjacency_result = check_adjacency(room_list[idx_1],room_list[idx_2],door_width,half_wall_width,share_idx_pair[share])
            if adjacency_result is not None:
                adjacency = [idx_1,idx_2,wall_idx,adjacency_result[0],adjacency_result[1]]
                adjacency_tabel.append(adjacency)

    new_wall_list = wall_list.copy()
    #检测全图连通性
    connected_component_list = [[idx] for idx in range(len(room_list))]
    for adj in adjacency_tabel:
        for idx,cc in enumerate(connected_component_list):
            if adj[0] in cc:
                a_idx = idx
            if adj[1] in cc:
                b_idx = idx
        if a_idx == b_idx:
            if np_random.rand()<extra_door_prop:
                new_door = place_door_wall(new_wall_list[adj[2]],adj[3],adj[4],half_wall_width, door_width, np_random)
                new_wall_list[adj[2]]['door'].append(new_door)
        else:
            connected_component_list[a_idx].extend(connected_component_list[b_idx])
            connected_component_list.pop(b_idx)
            new_door = place_door_wall(new_wall_list[adj[2]],adj[3],adj[4],half_wall_width, door_width, np_random)
            new_wall_list[adj[2]]['door'].append(new_door)
    # 非连通图，无法构造
    if len(connected_component_list)>1:
        return None
    return new_wall_list

def wall2fence(wall_list):
    basic_fence_prop = {}
    basic_fence_prop['color'] = [0, 0, 0]
    basic_fence_prop['anchor_x'] = 0
    basic_fence_prop['anchor_y'] = 0
    basic_fence_prop['rotation'] = 0
    basic_fence_prop['filled'] = True
    basic_fence_prop['close'] = True
    Fence_list = {}
    for wall in wall_list:
        x = wall['coord'][0]
        y = wall['coord'][1]
        direction = wall['direction']
        hww = wall['hww']
        l = wall['length']
        door_list = wall['door']
        start = 0
        if len(door_list)>0:
            for door in sorted(door_list,key = lambda x:x[0]):
                dc = door[0]
                dw = door[1]
                if direction is 0:
                    vertices = [[x-hww,y+start],[x+hww,y+start],[x+hww,y+dc],[x-hww,y+dc]]
                else:
                    vertices = [[x+start,y-hww],[x+start,y+hww],[x+dc,y+hww],[x+dc,y-hww]]
                wall_fence = basic_fence_prop.copy()
                wall_fence['vertices_x'] = [x for (x,y) in vertices]
                wall_fence['vertices_y'] = [y for (x,y) in vertices]
                Fence_list[len(Fence_list)] = wall_fence
                start = dc+dw

        if direction is 0:
            vertices = [[x-hww,y+start],[x+hww,y+start],[x+hww,y+l],[x-hww,y+l]]
        else:
            vertices = [[x+start,y-hww],[x+start,y+hww],[x+l,y+hww],[x+l,y-hww]]
        wall_fence = basic_fence_prop.copy()
        wall_fence['vertices_x'] = [x for (x,y) in vertices]
        wall_fence['vertices_y'] = [y for (x,y) in vertices]
        Fence_list[len(Fence_list)] = wall_fence
    return Fence_list

def random_fence(map_W,map_H,half_wall_width, car_R, door_width, room_number,np_random, max_dead_count = 1000):
    room_list, wall_list, placeable_area_list = random_room(map_W = map_W,
                                          map_H = map_H,
                                          half_wall_width = half_wall_width,
                                          car_R = car_R,
                                          door_width = door_width,
                                          room_number = room_number,
                                          np_random = np_random)

    wall_list = place_door(room_list,wall_list,half_wall_width,door_width,np_random)
    fence_dict = wall2fence(wall_list)
    return fence_dict, placeable_area_list

def render(wall_list, placeable_area_list = None):
    from . import rendering 
    viewer = rendering.Viewer(int(600),int(600))
    viewer.set_bounds(-1,+9,-1,+9)

    for wall in wall_list:
        x = wall['coord'][0]
        y = wall['coord'][1]
        direction = wall['direction']
        hww = wall['hww']
        l = wall['length']
        door_list = wall['door']
        start = 0
        if len(door_list)>0:
            for door in sorted(door_list,key = lambda x:x[0]):
                dc = door[0]
                dw = door[1]
                if direction is 0:
                    vertices = [[x-hww,y+start],[x+hww,y+start],[x+hww,y+dc],[x-hww,y+dc]]
                else:
                    vertices = [[x+start,y-hww],[x+start,y+hww],[x+dc,y+hww],[x+dc,y-hww]]  
                geom = rendering.make_polygon(vertices)
                viewer.add_geom(geom)
                start = dc+dw

        if direction is 0:
            vertices = [[x-hww,y+start],[x+hww,y+start],[x+hww,y+l],[x-hww,y+l]]
        else:
            vertices = [[x+start,y-hww],[x+start,y+hww],[x+l,y+hww],[x+l,y-hww]]
        geom = rendering.make_polygon(vertices)
        viewer.add_geom(geom)
    if placeable_area_list is not None:
        for area in placeable_area_list:
            [x,y,w,h] = area
            vertices = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
            geom = rendering.make_polygon(vertices)
            geom.set_color(1.0,0.0,0.0,alpha = 0.5)
            viewer.add_geom(geom)
    viewer.render(return_rgb_array = False)