import pygame
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
qTableHistory=[]
def qLearning():
    learning_rate=0.9
    discount=0.5
    random_explore=0.2
    qtable=pd.DataFrame(100, index=['up', 'down', 'left', 'right'], columns=[11,12,13,21,22,23,31,32,33,41,42,43])

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
                screen.fill((255,255,255))
            for block in grid:
                block.pointerDirection=qtable[block.id].idxmax()
                block.render(screen)
            pygame.display.update()
        qTableHistory.append(qtable)


SCREEN_WIDTH=800
SCREEN_HEIGHT=800

pygame.init()

screen=pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption("Q-Learning")

class Block():
    def __init__(self,xPos,yPos):
        self.xPos=xPos
        self.yPos=yPos

        self.blockWidth=200
        self.blockHeight=266

        self.blockRect=pygame.Rect(self.xPos,self.yPos,self.blockWidth,self.blockHeight)
        self.id=None

        self.center=(self.xPos+(self.blockWidth/2),self.yPos+(self.blockHeight/2))

        self.pointerDirection=""

    def render(self,screen):
        pygame.draw.line(screen,(0,0,0),(self.xPos,self.yPos),(self.xPos+self.blockWidth,self.yPos+self.blockHeight),1)
        pygame.draw.line(screen,(0,0,0),(self.xPos,self.yPos+self.blockHeight),(self.xPos+self.blockWidth,self.yPos),1)
        self.policyDirection(self.pointerDirection,screen)
        if self.id==41:
            pygame.draw.rect(screen,(0,255,0),self.blockRect)
        elif self.id==22 or self.id==42:
            pygame.draw.rect(screen,(255,0,0),self.blockRect)
        else:
            pygame.draw.rect(screen,(0,0,0),self.blockRect,1)
    
    def policyDirection(self,direction,screen):
        if direction=="":
            return
        if direction=="up":
            pygame.draw.polygon(screen,(0,255,0),[(self.xPos,self.yPos),(self.xPos+self.blockWidth,self.yPos),self.center]) 
        if direction=="down":
            pygame.draw.polygon(screen,(0,255,0),[(self.xPos,self.yPos+self.blockHeight),(self.xPos+self.blockWidth,self.yPos+self.blockHeight),self.center])
        if direction=="right":
            pygame.draw.polygon(screen,(0,255,0),[(self.xPos+self.blockWidth,self.yPos),(self.xPos+self.blockWidth,self.yPos+self.blockHeight),self.center])
        if direction=="left":
            pygame.draw.polygon(screen,(0,255,0),[(self.xPos,self.yPos),(self.xPos,self.yPos+self.blockHeight),self.center])

grid=[]
def updateGrid():
    ix=1
    for xPos in range(0,SCREEN_WIDTH,200):
        iy=1
        for yPos in range(0,SCREEN_HEIGHT-266,266):
            block=Block(xPos,yPos)
            block.id=ix*10+iy
            grid.append(block)
            iy+=1
        ix+=1
            

def main():
    counter=0
    isGameRunning=True
    updateGrid()
    qLearning()
    # print(qTableHistory[10])
    # print(qTableHistory[100])
    # print(qTableHistory[999])
    while isGameRunning:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isGameRunning=False
        screen.fill((255,255,255))
        for block in grid:
            block.pointerDirection=qTableHistory[counter][block.id].idxmax()
            block.render(screen)
        pygame.display.update()

        counter+=1
        if counter>=len(qTableHistory)-1:
            counter=len(qTableHistory)-1
            #counter=0
    
    return 0


if __name__=="__main__":
    main()