import gym
import numpy as np
import matplotlib.pyplot as plt

env=gym.make('MountainCar-v0')
env.reset()
'''print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)'''
LEARNING_RATE=0.1
DISCOUNT=0.95#reward_allowance
EPISODES=25000
SHOW_EVERY=2000
DISCRETE_OBS_SIZE=[20]*len(env.observation_space.high)
DISCRETE_OBS_WIN_SIZE=(env.observation_space.high-env.observation_space.low)/DISCRETE_OBS_SIZE
#print(DISCRETE_OBS_WIN_SIZE)
eps=0.5
START_EPSILON_DECAYING=1
END_EPSILON_DECAYING=EPISODES//2
epsilon_decay_value=eps/(END_EPSILON_DECAYING- START_EPSILON_DECAYING)


q_table=np.random.uniform(low=-2,high=0,size=(DISCRETE_OBS_SIZE+[env.action_space.n]))
#20x20x3

ep_rewards=[]
aggr_ep_rewards={'ep':[],'avg':[],'max':[],'min':[]}

def get_discrete_states(state):
	discrete_state=(state-env.observation_space.low)/DISCRETE_OBS_WIN_SIZE
	return tuple(discrete_state.astype(np.int))
for episode in range(EPISODES):
	episode_reward=0
	if episode%SHOW_EVERY==0:
		render=True
		print(episode)
	else:
		render=False

	discrete_state=get_discrete_states(env.reset())
	done=False

	#env.reset gives initial state

	while not done:
		if np.random.random()>eps:
			action=np.argmax(q_table[discrete_state])
		else:
			action=np.random.randint(0,env.action_space.n)
		new_state,reward,done,info=env.step(action) #new_state->position,velocity
		episode_reward+=reward
		new_discrete_state=get_discrete_states(new_state)

		if episode%2000==0:
			env.render()
		#print(new_state)
		#print(reward,new_state)
		if not done:
			max_future_q=np.max(q_table[new_discrete_state])
			current_q=q_table[discrete_state+(action, )]
			new_q=(1-LEARNING_RATE)*current_q +LEARNING_RATE*(reward+ DISCOUNT*max_future_q)
			q_table[discrete_state+(action,)]=new_q		

		elif new_state[0]>=env.goal_position:
			#env.render()
			print(f'Successful ascent @ {episode}')
			q_table[discrete_state+(action,)]=0
		discrete_state=new_discrete_state
	if END_EPSILON_DECAYING>=episode>=START_EPSILON_DECAYING:
		eps-=epsilon_decay_value
	ep_rewards.append(episode_reward)

	if not episode%SHOW_EVERY:
		average_reward=sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
		aggr_ep_rewards['ep'].append(episode)
		aggr_ep_rewards['avg'].append(average_reward)
		aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
		aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))	

		print(f'Episode:{episode} Avg:{average_reward} Min:{min(ep_rewards[-SHOW_EVERY:])} Max:{max(ep_rewards[-SHOW_EVERY:])}')
	

env.close()
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label='min')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label='max')
plt.legend(loc=4)
plt.show()