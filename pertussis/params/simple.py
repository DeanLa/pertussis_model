def collect_params(t, step=12):
    # [1] Neal Ferguson - A change in
    # [2] Headi Dangelis - Priming with...
    # [3] Meagan C F - Cost Effectiveness of Next gen...

    # rates
    beta = 4.301 * (1 / step)  # Force of infection - S stay at home [] daily
    lambda_ = beta
    gamma = (1/16) * (1 / step)  # Healing rate Asymptomatic [1] 16 daily
    omega = 3e-5 * (1 / step) # Loss of immunity [1] 3e-5 est yearly

    # Probabilites
    c = 0.95  # coverage
    aP = 0.95
    wP = 0.95

    return lambda_, gamma, omega


def collect_state0():
    # Compartments (State 0)
    S = 1 - 1e-6
    I = 1e-6
    R = 0

    # return S1, S2, Ia, Is, V, R, D
    return S, I, R