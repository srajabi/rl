

def epsilon_schedule(total_episodes, episode, min_epsilon):
    tenth_eps = total_episodes/10
    cur_eps = (tenth_eps - episode) / float(tenth_eps)
    return max(cur_eps, min_epsilon)
