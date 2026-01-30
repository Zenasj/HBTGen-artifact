import pickle
import tqdm
from torch.nn import MSELoss
import math
from collections import deque
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F




class neural_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.Linear(4,128)
        self.l2=nn.Linear(128,128)
        self.l3=nn.Linear(128,2)


    def forward(self,input):
        out=F.relu(self.l1(input))
        out=F.relu(self.l2(out))
        out=self.l3(out)
        return out

class ReplayBuffer():
    def __init__(self):
        self.max_len=10000
        self.capacity=0
        self.memory=deque([],maxlen=self.max_len)

    def push(self, transition):
        self.memory.append(transition)
        if self.capacity<self.max_len:
            self.capacity +=1


    def pull(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return self.capacity



def update_target(net1,net2):
    net1.load_state_dict(net2.state_dict())

def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)




def dqn(epsilon=0.9,alpha=1e-4,number_of_episodes=10000,gamma=0.99):
    env=gym.make("CartPole-v1")    # 4d states,   2 actions left right

    model=neural_net()
    target=neural_net()
    update_target(target,model)

    replay_buffer=ReplayBuffer()

    number_of_episodes_seen=0
    total_number_of_frames_seen=0

    criterion=MSELoss()
    optim=torch.optim.Adam(params=model.parameters(),lr=alpha)

    for i in tqdm.tqdm(range(number_of_episodes)):
        number_of_episodes_seen+=1
        state,_=env.reset()
        ep_done=False

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        while not ep_done:
            action_values=model(state)

            r = random.random()
            if r>epsilon:
                action=torch.argmax(action_values).item()  #the greedy action
            else:
                action=random.choice([0,1])  #the epsilon greedy action

            next_state,reward,terminated,truncated,_=env.step(action)
            total_number_of_frames_seen+=1

            #update epsilon
            epsilon = 0.05 + (0.9 - 0.05) * math.exp(-1 * total_number_of_frames_seen / 1000)

            ep_done=terminated or truncated


            next_state=torch.tensor(next_state,dtype=torch.float32).unsqueeze(0)
            reward=torch.tensor([reward],dtype=torch.float32)
            done=torch.tensor([ep_done])
            action=torch.tensor([action])

            replay_buffer.push([state,action,reward,next_state,done])

            state=next_state


            if len(replay_buffer)>=128:
                batch=replay_buffer.pull(128)
                # this will look like [[s,a,r,s'],[s,a,r,s'],...]

                batch = list(zip(*batch))
                #converts it to [(s,s,s,,..),(a,a,a,,...),(r,r,r...),(s',s',s'...),(done,done,done...),(w,w,w,...)]

                state_batch=torch.concat(batch[0],dim=0)
                action_batch=torch.concat(batch[1],dim=0).type(torch.int64)
                scalar_reward_batch=torch.concat(batch[2],dim=0)
                next_state_batch=torch.concat(batch[3],dim=0)
                done_batch=torch.concat(batch[4],dim=0).type(torch.bool)

                action_values=model(state_batch)
                action_values=action_values.gather(1,action_batch.unsqueeze(1))
                # note: needed to unsqueeze because action_batch needs to have same shape as model outputs in gather



                next_state_action_values = target(next_state_batch)
                next_state_action_values = next_state_action_values.gather(1,next_state_action_values.argmax(dim=1).unsqueeze(1))


                y = scalar_reward_batch.unsqueeze(1) + ~done*gamma*next_state_action_values


                loss=criterion(action_values, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                soft_update(target, model)
                ## the model is now updated



        #episode has terminated



        if number_of_episodes_seen%50==0:
            #run an eval episode

            state,_= env.reset()
            ep_done=False
            total_return = 0.
            state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)

            while not ep_done:
                action=model(state).argmax().item()
                next_state,reward,terminated,truncated,_=env.step(action)

                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

                total_return+=reward

                state=next_state

                ep_done=terminated or truncated



            print("\n_______________________\n episode:",number_of_episodes_seen,"\nnumber of frames seen:",total_number_of_frames_seen,"\nscalarised reward: ",total_return,"\n_______________________")

    return model

model=dqn()

a=open("model","wb")
pickle.dump(model,a)
a.close()