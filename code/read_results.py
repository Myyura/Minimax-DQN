import os
import json
import matplotlib.pyplot as plt
import numpy as np

def show_result(jsonpath_maxmean, jsonpath_std, title: str, filename: str, max_step: int):
    with open(jsonpath_maxmean, 'r') as f:
        scores_maxmean = json.load(f)
    with open(jsonpath_std, 'r') as f:
        scores_std = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 8))

    x1 = list(scores_maxmean.keys())
    y1 = list(scores_maxmean.values())
    x1 = [int(x) for x in x1]
    y1 = [[float(yy) for yy in y] for y in y1]
    y1_mean = [np.average(y1[i]) for i in range(len(y1))]
    y1_max = np.max(y1, axis=1)
    y1_min = np.min(y1, axis=1)

    x2 = list(scores_std.keys())
    y2 = list(scores_std.values())
    x2 = [int(x) for x in x2]
    y2 = [[float(yy) for yy in y] for y in y2]
    y2_mean = [np.average(y2[i]) for i in range(len(y2))]
    y2_max = np.max(y2, axis=1)
    y2_min = np.min(y2, axis=1)

    if 'MountainCar' in title:
        yaxis_min = -200
        min_score = -110
    if 'CartPole' in title:
        yaxis_min = 0
        min_score = 195 / 200 * 500
    if 'LunarLander' in title:
        yaxis_min = -200
        min_score = 200
    if 'Acrobot' in title:
        yaxis_min = -500
        min_score = -100

    if 'LunarLander' in title:
        ax.set_ylim(-400, 300)

    ax.plot(x1, y1_mean, 'o-', color='r', label = 'Maxmean-DQN', alpha=0.7)
    plt.fill_between(x1, y1_min, y1_max, color='r', alpha=0.2)

    ax.plot(x2, y2_mean, '*-', color='grey', label = 'Standard-DQN', alpha=0.7)
    plt.fill_between(x2, y2_min, y2_max, color='grey', alpha=0.2)

    # ax.set_xticks([max_step // 10 * i for i in range(1, 11)])
    # ax.set_xticklabels([max_step // 10 * i for i in range(1, 11)])
    ax.set_xlabel('Step')
    ax.set_ylabel('Score')
    ax.legend(loc=4)
    ax.set_title(title)
    plt.savefig(os.path.join('..', 'result',  filename + '.png'))
    
    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 8))
    max_score = max(np.max(y1_mean), np.max(y2_mean))
    print(max_score, type(max_score))

    # max score
    b1 = [yaxis_min]
    b2 = [yaxis_min]
    ms1 = yaxis_min
    ms2 = yaxis_min
    for i in range(min(len(y1_mean), len(y2_mean))):
        if y1_mean[i] > ms1:
            ms1 = y1_mean[i]
        if y2_mean[i] > ms2:
            ms2 = y2_mean[i]
        b1.append(ms1)
        b2.append(ms2)

    if 'LunarLander' in title:
        x1_ms = [64 * i for i in range(1, len(b1) + 1)]
        x2_ms = [64 * i for i in range(1, len(b2) + 1)]
    else:
        x1_ms = [32 * i for i in range(1, len(b1) + 1)]
        x2_ms = [32 * i for i in range(1, len(b2) + 1)]

    ax.plot(x1_ms, b1, '-', color='r', label = 'Maxmean-DQN')
    ax.plot(x2_ms, b2, '-', color='grey', label = 'Standard-DQN')

    # std_low = ms2 * 0.9
    # std_high = ms2 * 1.1
    # if std_low > std_high:
    #     std_low, std_high = std_high, std_low

    # ax.hlines(y=min_score, xmin=0, xmax=x2_ms[-1], color=['k'], linestyles=['--', '--'])
    # ax.fill_between([0, x2_ms[-1]], std_low, std_high, facecolor='k', alpha=0.2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Max Score')
    ax.legend(loc=4)
    ax.set_title(title)
    plt.savefig(os.path.join('..', 'result',  filename + '_max_score.png'))

    plt.clf()

    fig, ax = plt.subplots(figsize=(8, 8))
    y1_sum = 0
    y1_sumave = []
    for i, score in enumerate(y1_mean):
        y1_sum += score
        y1_sumave.append(y1_sum / (i + 1))
    
    y2_sum = 0
    y2_sumave = []
    for i, score in enumerate(y2_mean):
        y2_sum += score
        y2_sumave.append(y2_sum / (i + 1))

    if 'LunarLander' in title:
        x1_sumave = [64 * i for i in range(1, len(y1_sumave) + 1)]
        x2_sumave = [64 * i for i in range(1, len(y2_sumave) + 1)]
    else:
        x1_sumave = [32 * i for i in range(1, len(y1_sumave) + 1)]
        x2_sumave = [32 * i for i in range(1, len(y2_sumave) + 1)]
    
    ax.plot(x1_sumave, y1_sumave, '-', color='r', label = 'Maxmean-DQN')
    ax.plot(x2_sumave, y2_sumave, '-', color='grey', label = 'Standard-DQN')
    # ax.fill_between(x2, y2_min, y2_max, color='b', alpha=0.2)
    # ax.fill_between(x1, y1_min, y1_max, color='g', alpha=0.2)
    if 'LunarLander' in title:
        ax.set_ylim((-200, 300))
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Score')
    ax.legend(loc=4)
    ax.set_title(title)
    plt.savefig(os.path.join('..', 'result',  filename + '_mean_score.png'))

if __name__ == '__main__':
    show_result(
        '/data/zhangzhe/Minimax-DQN/result/1_correct_cartpole_minimax_42_100000_10000.json', 
        '/data/zhangzhe/Minimax-DQN/result/1_correct_cartpole_standard_42_100000_10000.json',
        'CartPole-v1',
        'final_cartpole_42_100000',
        100000)

    show_result(
        '/data/zhangzhe/Minimax-DQN/result/2_correct_mountaincar_minimax_42_500000_50000.json', 
        '/data/zhangzhe/Minimax-DQN/result/2_correct_mountaincar_standard_42_500000_50000.json',
        'MountainCar-v0',
        'final_mountaincar_42_500000',
        500000)

    # show_result(
    #     '/data/zhangzhe/Minimax-DQN/result/tt_mountaincar_minimax_42_500000.json', 
    #     '/data/zhangzhe/Minimax-DQN/result/tt_mountaincar_standard_42_500000.json',
    #     'MountainCar-v0',
    #     'tt_mountaincar_42_500000',
    #     500000)

    show_result(
        '/data/zhangzhe/Minimax-DQN/result/1_correct_lunarlander50_minimax_42_500000_50000.json', 
        '/data/zhangzhe/Minimax-DQN/result/1_correct_lunarlander50_standard_42_500000_50000.json',
        'LunarLander-v2',
        'final_lunarlander_42_500000',
        500000)

    show_result(
        '/data/zhangzhe/Minimax-DQN/result/2_correct_acrobot_minimax_42_20000_2000.json', 
        '/data/zhangzhe/Minimax-DQN/result/2_correct_acrobot_standard_42_20000_2000.json',
        'Acrobot-v1',
        'final_acrobot_42_20000',
        20000)