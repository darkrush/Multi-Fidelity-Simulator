from MF_Sim.HF_Sim import random_map, simulator,core
import time

half_wall_width = 0.05
car_R = 0.2
door_width = 0.8
room_number = 5
agent_number = 1
room_list, wall_list, placeable_area_list = random_map.random_room(map_W = 8,
                                              map_H = 8,
                                              half_wall_width = half_wall_width,
                                              car_R = car_R,
                                              door_width = door_width,
                                              room_number = room_number)
wall_list = random_map.place_door(room_list,wall_list,half_wall_width,door_width)
fence_dict = random_map.wall2fence(wall_list)

agent_dict = simulator.random_agent(placeable_area_list, agent_number)
world = core.World(agent_dict,fence_dict)
world.reset()
action = core.Action()
action.ctrl_phi = 1.0
action.ctrl_vel = 1.0
    
while True:
    action.ctrl_phi = -action.ctrl_phi
    world.set_action([action])
    world.step()
    world.render()