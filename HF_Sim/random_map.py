from .basic import AgentState,Action,Observation,Fence
import numpy as np

def place_wall(cord,car_width):
    first_direction = np.random.randint(2)
    placed_cord = -1
    if cord[2+first_direction] > 2*car_width:
        placed_cord = first_direction
    elif cord[3-first_direction] > 2*car_width:
        placed_cord = 1-first_direction
    if placed_cord > -1:
        wall_cord = np.random.rand()*(cord[2+placed_cord]-2*car_width)+car_width
        wall = [placed_cord,wall_cord]
        first_cord = cord.copy()
        first_cord[2+placed_cord] = wall_cord
        second_cord = cord.copy()
        second_cord[placed_cord] = second_cord[placed_cord] + wall_cord
        second_cord[2+placed_cord] = second_cord[2+placed_cord] - wall_cord
        return [wall,first_cord,second_cord]
    else:
        return None

def random_map(map_W, map_H, pixel_size, car_width, wall_number, agent_number):
    placed_wall_number=0
    root_room = {'cord':[-map_W/2,-map_H/2,map_W,map_H], 'wall':None, 'children':None, 'parent':None}
    dead_count = 1000
    leaf_room_list = [root_room]
    while placed_wall_number<wall_number:
        placed = False
        for current_id in range(len(leaf_room_list)):
            wall = place_wall(leaf_room_list[current_id]['cord'],car_width)
            if wall is not None:
                placed = True
                break
        if placed:
            print(leaf_room_list)
            first_children = {'cord':wall[1], 'wall':None, 'children':None, 'parent':leaf_room_list[current_id]}
            second_children = {'cord':wall[2], 'wall':None, 'children':None, 'parent':leaf_room_list[current_id]}
            leaf_room_list[current_id]['wall'] = wall[0]
            leaf_room_list[current_id]['children'] = [first_children,second_children]
            leaf_room_list.pop(current_id)
            leaf_room_list.append(first_children)
            leaf_room_list.append(second_children)
            placed_wall_number = placed_wall_number + 1
        else:
            raise Exception("No leaf_room can place wall")
        
        dead_count = dead_count-1
        if dead_count==0:
            raise Exception("failed")
    return root_room

def render(root_room):
    from . import rendering 
    viewer = rendering.Viewer(int(400),int(400))
    viewer.set_bounds(-5,+5,-5,+5)
    next_node = [root_room]
    wall_list = []
    while len(next_node)>0:
        if next_node[0]['wall'] is not None:
            wall_list.append((next_node[0]['cord'],next_node[0]['wall']))
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
    for (cord,wall) in wall_list:
        x = cord[0]
        y = cord[1]
        w = cord[2]
        h = cord[3]
        wc = wall[1]
        wall_width = 0.05
        if wall[0] is 0:
            vertices = [[x+wc-wall_width,y],[x+wc+wall_width,y],[x+wc+wall_width,y+h],[x+wc-wall_width,y+h]]
        else:
            vertices = [[x,y+wc-wall_width],[x,y+wc+wall_width],[x+w,y+wc+wall_width],[x+w,y+wc-wall_width]]
        geom = rendering.make_polygon(vertices)
        viewer.add_geom(geom)
    viewer.render(return_rgb_array = False)
#if __name__=='__main__':
#    print(place_wall())