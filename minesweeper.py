import config as cfg
import numpy as np

class MineSweeper:
    def __init__(self) -> None:
        """
        minesweeper

        10X10 grid
        10 bombs randomly placed

        -1 for bombs
        0 for no number
        1-8 for cells by bomb
        
        need to set the number to each box


        """
        self.num_cells = cfg.num_cells
        self.num_mines = cfg.num_mines
        self.reset()


    def set_init_board(self):
        l = np.zeros((self.num_cells,self.num_cells), dtype=int)
        return l
    
    def reset(self):
        self.board = self.set_init_board()
        self.solution = self.set_init_board()
        self.set_mines()
        self.set_cell_nums()
    
    def get_board(self):
        return self.board
    
    def set_mines(self):
        row = np.random.randint(self.num_cells, size=self.num_mines)
        col = np.random.randint(self.num_cells, size=self.num_mines)
        for i,j in zip(row,col):
            self.board[i][j] = -1

    def set_cell_nums(self):
        pass
    

if __name__=='__main__':
    m = MineSweeper()
    print(m.get_board())
    m.set_mines()
    print(m.get_board())