import numpy as np
import pandas as pd
import config as cfg

class Sudoku:
    def __init__(self) -> None:
        """
            self.init_board  - 9x9 initial fixed board 
            self.current_board - 9x9 board thats played on
            self.answer_board - 9x9 solution to init board
            self.dead - 0 when alive and 1 if mistake happened
            self.result - 1 only when solution == board
        """
        self.init_board = self.set_init_board()
        self.board = self.set_init_board()
        self.solution = self.set_init_board()

    def reset(self,b,a):
        for i in range(len(b)):
            r = i//9
            c = i%9

            self.init_board[r][c] = self.board[r][c] = int(b[i])
            self.solution[r][c] = int(a[i])
        
        return self.board
    
    def get_mask(self):
        return self.init_board

    def set_init_board(self):
        l = np.zeros((9,9), dtype=int)
        return l

    #   converts string of 81 numbers to 9x9 int numpy array 

    def convert_1d(self,board):
        return board.flatten()

    def get_board(self):
        return self.board

    #   checks if board == solution return true is correct else false
    def check_solution(self):
        if (self.board == self.solution).all():
            return 1
        else:
            return 0

    def get_rcn_action(self,cell):
        n = cell//81 +1
        l = cell%81
        r = l//9
        c = l%9
        return r, c, n
    
    #   adds number to row and col
    def add_num(self,row,col,num):
        self.board[row][col] = num

    #   checks num for collisions
    def check_num_collision(self,row,col,num):
        flag = 1
        r = self.board[row,:]
        c = self.board[:,col]
        g = self.board[(row//3)*3:(row//3)*3+3,(col//3)*3:(col//3)*3+3]

        if num in r:
            flag = 0

        if num in c:
            flag = 0

        if num in g.flatten():
            flag = 0

        return flag

    #   checks actions for collisions
    def check_action(self,action):
        row, col, num = self.get_rcn_action(action)

        if self.init_board[row][col] == 0:
            if self.check_num_collision(row,col,num):
                self.add_num(row,col,num)
                return cfg.reward_no_collision
            else:
                return cfg.reward_collision_num
        else:
            return cfg.reward_filled_cell
        
    def step(self, action):
        reward = self.check_action(action)
        if self.check_solution():
            reward = cfg.reward_correct_end

        observation_ = self.get_board()

        if reward<0:
            done = True
        else:
            done = False
            
        return observation_, reward, done
    
    def check_acc(self):
        acc_count = 0
        total_count = 0
        for r in range(9):
            for c in range(9):
                if self.init_board[r][c]==0:
                    total_count +=1
                    if self.board[r][c] == self.solution[r][c]:
                        acc_count +=1 
        
        return acc_count/total_count *100

if __name__=='__main__':
    path = 'F:\\sudoku_RL\\sudoku_10k_1.csv'

    df = pd.read_csv(path)
    board = df.iloc[0, 0]
    ans = df.iloc[0, 1]
    s = Sudoku()
    s.reset(board,ans)
