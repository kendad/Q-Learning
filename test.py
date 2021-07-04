import pandas as pd
import random

class Game():
    rewards=None
    positionCol=None
    positonRow=None

    def __init__(self,startCol=1,startRow=1):
        self.rewards=pd.DataFrame({1:[0,1,2,3,4], 2:[1,2,3,4,5], 3:[2,3,4,5,6], 4:[3,4,5,6,7], 5:[4,5,6,7,8]}, index={1,2,3,4,5})
        self.positionCol=startCol
        self.positonRow=startRow

    def move(self,direction):
        reward=0
        end=False
        if direction=="up":
            self.positonRow-=1
        elif direction=="down":
            self.positonRow+=1
        elif direction=="left":
            self.positionCol-=1
        else:
            self.positionCol+=1
        
        #check if we lost
        if self.positonRow<1 or self.positionCol>5 or self.positionCol<1 or self.positonRow>5:
            end=True
            reward=-1000
        #check if we have reached the goal
        elif self.positionCol==5 and self.positonRow==5:
            end=True
            reward=1000
        #if agent in normal state then get reward from rewards table
        else:
            end=False
            reward=self.rewards[self.positionCol][self.positonRow]
        
        #return reward and end of the game indicator
        return (reward,end)



#states are in COL and actions in ROW
learning_rate=0.9
discount=0.5
random_explore=0.2
qtable=pd.DataFrame(100, index=['up', 'down', 'left', 'right'], columns=[11,12,13,14,15,21,22,23,24,25,31,32,33,34,35,41,42,43,44,45,51,52,53,54,55])


for i in range(1000):
    game=Game()
    end_of_game=False

    while not end_of_game:
        #get current state
        current_state=(game.positionCol*10)+game.positonRow
        #select action with max rewards
        max_reward_action=qtable[current_state].idxmax()
        #promote exploration
        if random.random()<random_explore:
            max_reward_action=qtable.index[random.randint(0,3)]
        #play the game with that action
        reward,end_of_game=game.move(max_reward_action)

        if end_of_game:
            qtable.loc[max_reward_action,current_state]=reward
        else:
            #if game not ended, then get the next states max Q-value(optimal future value)
            optimal_future_value=qtable[(game.positionCol*10)+game.positonRow].max()
            #multiply this with the sicount factor
            discounted_optimal_future_value=discount*optimal_future_value
            #learned value
            learned_value=reward+discounted_optimal_future_value
            #updated the Qvalue for the action taken
            qtable.loc[max_reward_action,current_state]=(1-learning_rate)*qtable[current_state][max_reward_action]+ learning_rate*learned_value


print(qtable)