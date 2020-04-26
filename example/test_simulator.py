from MF_Sim.HF_Sim import simulator,utils

policy_args={}
policy_args['max_phi'] = 3.14159/6.0
policy_args['l'] = 0.3
policy_args['dist'] = 0.1

n_policy = utils.naive_policy(policy_args)

map_W = 4.0
map_H = 10.0
half_wall_width = 0.05
car_R = 0.2
door_width = 0.8
room_number = 4
agent_number = 3
near_dist = 2.0
Env = simulator.Full_env(map_W ,map_H,room_number,door_width,half_wall_width,agent_number,near_dist)
obs = Env.reset()
while True:
    action_list = n_policy.inference(obs,[])
    obs,reward,done,info = Env.step(action_list)
    #print('************')
    #print(obs)
    #print(reward)
    Env.render()