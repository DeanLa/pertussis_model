def collect_params():
    #### Params ####

    # rates
    mu = 1 / (75 * 365)   # Births
    aP = 0.1
    wP = 0.1
    beta = 520 / 365    # Force of infection
    gamma_a = 1 / 7   # Healing rate Asymptomatic
    gamma_s = 1 / 7   # Healing rate Symptomatic
    omega = 1 / 7
    sigma =  0.25 # Change to be symptomatic
    nu = mu
    return locals()

def collect_state0():
    # Compartments (State 0)
    S = 0.1
    Ia = 1e-4
    Is = 0
    V = 0
    R = 1 - S - Ia - Is - V

    return locals()