from agent import Agent
from sudoku import Sudoku
import config as cfg
from plot import plot

import numpy as np
import pandas as pd

from tqdm import tqdm


def train(df):
    env = Sudoku()
    num_games = cfg.num_games
    load_checkpoint = False

    agent = Agent(gamma=cfg.gamma, epsilon=cfg.epsilon, lr=cfg.lr, mem_size=cfg.mem_size, 
                  eps_min=0.01,batch_Size=cfg.batch_size, eps_dec=1e-5, replace=1000)
    
    if load_checkpoint:
        agent.load_models()

    scores, avg_scores = [],[]
    eps_history = []
    rounds, avg_rounds = [], []
    accs, avg_accs = [],[]
#    losses, avg_losses = [], []

    idx = 0
    pbar = tqdm(range(num_games),ncols=150)
    for i in pbar:
        pbar.set_description(f"Episode {i}")
        if idx == len(df)-1:
            idx = 0
        if i>cfg.num_same_game and i%cfg.num_same_game ==0:
            idx +=1
        done = False
        board = df.iloc[idx, 0]
        ans = df.iloc[idx, 1]
        observation = env.reset(board,ans)
        score = max_score = 0
        round = max_round = 0
        
        while not done:
            round +=1
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            #if not done:
            #    reward += round
            score += reward
            agent.store_transition(observation,action,reward,observation_,int(done))
            observation = env.get_board()
            if round>100:
                done = True
            
        agent.learn()

        max_round = max(max_round,round)
        max_score = max(max_score,score)

        acc = env.check_acc()
        accs.append(acc)
        avg_acc = np.mean(accs[-1000:])
        avg_accs.append(avg_acc)

        scores.append(score)
        avg_score = np.mean(scores[-1000:])
        avg_scores.append(avg_score)

        rounds.append(round)
        avg_round = np.mean(rounds[-1000:])
        avg_rounds.append(avg_round)

#        losses.append(loss/round)
#        avg_loss = np.mean(losses[-1000:])
#        avg_losses.append(avg_loss)

        
        #print(f"episode:{i}, avg_score:{avg_score:>0.5f}, avg_round:{avg_round:>0.5f}, avg_acc:{avg_acc:>0.5f}")
        pbar.set_postfix_str(f" avg_score:{avg_score:>0.5f}, avg_round:{avg_round:>0.5f}, avg_acc:{avg_acc:>0.5f}, epsilon:{agent.epsilon:>0.7f}")

        if i>1 and i%10000 == 0:
            print(f"max_round:{max_round}, max_score:{max_score} ")
            agent.save_models(i)

        eps_history.append(agent.epsilon)

    plot(scores,avg_scores,'scores')
    plot(rounds,avg_rounds,'rounds')
    plot(accs,avg_accs,'accuracy')
#    plot(losses,avg_losses,'loss')

def test(df):
    env = Sudoku()
    num_games = len(df)-1
    load_checkpoint = True

    agent = Agent(gamma=cfg.gamma, epsilon=0.0, lr=cfg.lr, mem_size=10000, 
                  eps_min=0.01,batch_Size=64, eps_dec=1e-3, replace=100)
    
    if load_checkpoint:
        agent.load_models()

    scores, avg_scores = [],[]
    rounds, avg_rounds = [], []
    accs, avg_accs = [],[]

    pbar = tqdm(range(num_games),ncols=150)
    for i in pbar:
        pbar.set_description(f"Test {i}")
        done = False
        board = df.iloc[i, 0]
        ans = df.iloc[i, 1]
        observation = env.reset(board,ans)
        score = 0
        round = 0
        while not done:
            round +=1
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward
            observation = env.get_board()
        
        acc = env.check_acc()
        accs.append(acc)
        avg_acc = np.mean(accs[-1000:])
        avg_accs.append(avg_acc)

        scores.append(score)
        avg_score = np.mean(scores[-1000:])
        avg_scores.append(avg_score)

        rounds.append(round)
        avg_round = np.mean(rounds[-1000:])
        avg_rounds.append(avg_round)

        
        #print(f"test board:{i}, score:{score:>0.5f}, rounds:{round:>0.5f}, accs:{acc:>0.5f}")
        pbar.set_postfix_str(f" avg_score:{avg_score:>0.5f}, avg_round:{avg_round:>0.5f}, avg_acc:{avg_acc:>0.5f}")

    plot(scores,avg_scores,'scores')
    plot(rounds,avg_rounds,'rounds')
    plot(accs,avg_accs,'accuracy')

if __name__=='__main__':
    path = cfg.data_path
    path_test = cfg.data_path_test
    df_train = pd.read_csv(path,header=None)
    df_test = pd.read_csv(path_test,header=None)
    train(df_train)
    #test(df_test)              need to set it up beacuse of changes to file structure
