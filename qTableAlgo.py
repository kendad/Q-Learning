import pandas as pd
import random

class Game():
    def __init__(self,startCol=1,startRow=1):
        self.rewards=pd.DataFrame({1:[-0.5,-0.5,-0.5],2:[-0.5,-10,-0.5],3:[-0.5,-0.5,-0.5],4:[1,-10,-0.5]},index={1,2,3})
        self.positionCol=startCol
        self.positionRow=startRow
        # print(self.rewards[4][1])#col,row
    
    def move(self,direction):
        reward=0
        end=False
        if direction=="up":
            self.positionRow-=1
        elif direction=="down":
            self.positionRow+=1
        elif direction=="left":
            self.positionCol-=1
        else:
            self.positionCol+=1
        
        #check if we lost the game
        #check if we lost
        if self.positionRow<1 or self.positionCol>4 or self.positionCol<1 or self.positionRow>3:
            end=True
            reward=-10
        #check if we won the game
        elif self.positionCol==4 and self.positionRow==1:
            end=True
            reward=self.rewards[4][1]
        else:
            end=False
            reward=self.rewards[self.positionCol][self.positionRow]
        
        return (reward,end)

#Rewards Table
#      1     2    3     4
# 1 -0.5  -0.5 -0.5   1.0
# 2 -0.5 -10.0 -0.5 -10.0
# 3 -0.5  -0.5 -0.5  -0.5
# #####
#QTable
#         11   12   13   21   22   23   31   32   33   41   42   43
# up     100  100  100  100  100  100  100  100  100  100  100  100
# down   100  100  100  100  100  100  100  100  100  100  100  100
# left   100  100  100  100  100  100  100  100  100  100  100  100
# right  100  100  100  100  100  100  100  100  100  100  100  100



learning_rate=0.9
discount=0.5
random_explore=0.2
qtable=pd.DataFrame(100, index=['up', 'down', 'left', 'right'], columns=[11,12,13,21,22,23,31,32,33,41,42,43])
qTableHistory=[]

for i in range(1000):
    game=Game()
    end_of_game=False

    while not end_of_game:
        #get current State
        current_state=(game.positionCol*10)+game.positionRow
        max_reward_action=qtable[current_state].idxmax()

        if random.random()<random_explore:
            max_reward_action=qtable.index[random.randint(0,3)]

        #play the game    
        reward,end_of_game=game.move(max_reward_action)

        if end_of_game:
            qtable.loc[max_reward_action,current_state]=reward
        else:
            optimal_future_value=qtable[(game.positionCol*10)+game.positionRow].max()
            discounted_optimal_future_value=discount*optimal_future_value
            learned_value=reward+discounted_optimal_future_value
            qtable.loc[max_reward_action,current_state]=(1-learning_rate)*qtable[current_state][max_reward_action]+ learning_rate*learned_value
            qTableHistory.append(qtable)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(qtable)
print("############")
print(qTableHistory[1])
