

def expected(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def elo(old_elo, exp_elo, score, k=20):
    return old_elo + k * (score - exp_elo)
