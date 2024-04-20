# model network
lr = 0.0001
hid_channel = 128
num_residuals = 10

# rewards
reward_filled_cell = -1
reward_collision_num = -1
reward_no_collision = 1
reward_correct_end = 100

# manager
mem_size = 100000
batch_size = 64
gamma = 0.6
epsilon = 0.7

data_path = 'F:\\sudoku_ai\\dataset\\sudoku_100k_1.csv'
data_path_test = 'F:\\sudoku_ai\\dataset\\sudoku_10k_1.csv'
chkpt_dir = 'F:\\sudoku_ai\\checkpoints'

num_games = 100000
num_same_game = 1000000

# minesweeper
num_cells = 10
num_mines = 10