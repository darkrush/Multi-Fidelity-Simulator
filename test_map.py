from HF_Sim import random_map
half_wall_width = 0.2
car_R = 0.2
door_width = 0.8
room_number = 5
room_list, wall_list,placeable_area_list = random_map.random_room(map_W = 8,
                                              map_H = 8,
                                              half_wall_width = half_wall_width,
                                              car_R = car_R,
                                              door_width = door_width,
                                              room_number = room_number)
#print('*********************')
#for room in room_list:
#    print(room)
#print('*********************')
#for wall in wall_list:
#    print(wall)
#print('*********************')
new_wall_list = random_map.place_door(room_list,wall_list,half_wall_width,door_width)
#for wall in new_wall_list:
#    print(wall)
#print('*********************')
random_map.render(new_wall_list,placeable_area_list)
fence_list = random_map.wall2fence(wall_list)
for n,fence in fence_list.items():
    print(fence)
while True:
    pass