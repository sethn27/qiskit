import gym
import numpy as np
import matplotlib.pyplot as plt

from ppo_agent_torch import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    N=20
    batch_size = 5
    n_epochs =  4
    alpha = 0.0003
    agent = Agent(n_actions = env.action_space.n, batch_size=batch_size, alpha = alpha, n_epochs = n_epochs, input_dims = env.observation_space.shape)
    n_games = 1000

    figure_file = ' plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []
    avg_score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0

        while not done:
            #env.render()
            #print('obs 1 is',observation)
            action, prob, val = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)[0:4]

            #print('obs 2 is',observation_,reward,done,info)

            n_steps += 1
            score += reward

            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N ==0:
                agent.learn()
                learn_iters += 1
            observation = observation_

            if score >= 500:
                break

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            #agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
        
        if i >= 600 and avg_score == 500:
            break
        
    x = [i+1 for i in range(len(score_history))]
    #plot_learning_curve(x, score_history, figure_file)

    plt.plot(x,score_history, label='Score')
    plt.plot(x,avg_score_history, label='Average')
    plt.show()