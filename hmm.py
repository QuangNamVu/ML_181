import numpy as np

m = np.array([
    #   |party| TV |Pub|Study
    [.05, .7, .05, .2],  # Party
    [.1, .4, .3, .2],  # TV
    [.1, .6, .05, .25],  # Pub
    [.25, .3, .4, .05],  # Study
])

o = np.array([
    # |Tired| Hung | Scared |Fine
    [.3, .4, .2, .1],  # Party
    [.2, .1, .2, .5],  # TV
    [.4, .2, .1, .3],  # Pub
    [.3, .05, .3, .35],  # Study
])
#                   |Tired| Hung | Scared |Fine
pi_start = np.array([.25, .25, .25, .25])

dict_names_trans_mat = {"Party": 0, "TV": 1, "Pub": 2, "Study": 3}

dict_feel = {"Tired": 0, "Hung": 1, "Scared": 2, "Fine": 3}


# 1, P(Par, TV, Pub, Tired, Scared, Fine)
def prop_state_feel(lst_state, lst_feel):
    # p0 = start [[Par]; Par -> TV; TV ->  Pub
    # p0 *= Tired, Scared, Fine
    start_state = dict_names_trans_mat[lst_state[0]]  # start = Party => 0
    start_feel = dict_feel[lst_feel[0]]  # Tired => 0
    p = pi_start[start_state] * o[start_state][start_feel]

    for current_idx in range(1, len(lst_state)):
        prev_state_idx = dict_names_trans_mat[lst_state[current_idx - 1]]
        current_state_idx = dict_names_trans_mat[lst_state[current_idx]]
        current_feel_idx = dict_feel[lst_feel[current_idx]]

        p *= m[prev_state_idx][current_state_idx]
        p *= o[prev_state_idx][current_feel_idx]

    return p


lst_state = ["Party", "TV", "Pub"]
lst_feel = ["Tired", "Scared", "Fine"]

p_1 = prop_state_feel(lst_state, lst_feel)
print("1, P(Par, TV, Pub, Tired, Scared, Fine)")
print(p_1) # 2.625*10^(-6)


# 2, find P(Hung , Scared, Tired)

def prop_hidden_state_feel(lst_obs):
    # s1 = start_state .* o[Hung]
    n_layer = len(lst_feel)
    s = np.zeros(shape=(len(pi_start), n_layer))  # shape = 4x3
    o_start_idx = dict_feel[lst_obs[0]]
    s[:, 0] = np.multiply(pi_start, o[:, o_start_idx])

    # s2 = M'.dot(s1) .* o[Scared] and so on
    for ilayer in range(1, n_layer):
        o_start_idx = dict_feel[lst_feel[ilayer]]
        s[:, ilayer] = m.transpose().dot(s[:, ilayer - 1])
        s[:, ilayer] = np.multiply(s[:, ilayer], o[:, o_start_idx])

    return sum(s[:, n_layer - 1])


p_2 = prop_hidden_state_feel(["Hung", "Scared", "Tired"])
print("2, P(Hung , Scared, Tired) \n",p_2)

log_m = np.log2(m)

log_o = np.log2(o)

log_pi_start = np.log2(pi_start)


# 3, find state in sequence observes HungOver, Scared , Tired

def viterbi(lst_obs, trace_array=[]):
    n_cols = len(lst_feel)
    n_rows = len(pi_start)

    dynamic_prog_table = np.zeros(shape=(n_rows, n_cols))

    o_start_idx = dict_feel[lst_obs[0]]

    dynamic_prog_table[:, 0] = log_pi_start + log_o[:, o_start_idx]

    for icol in range(1, n_cols):
        # get max row from previous

        max_idx_prev_row = np.argmax(dynamic_prog_table[:, icol - 1])

        # trace
        trace_array.append(
            list(dict_names_trans_mat.keys())[list(dict_names_trans_mat.values()).index(max_idx_prev_row)])

        p_prev = dynamic_prog_table[max_idx_prev_row, icol - 1]
        o_idx = dict_feel[lst_obs[icol]]
        dynamic_prog_table[:, icol] = p_prev + log_m[max_idx_prev_row, :] + log_o[:, o_idx]

    max_idx_last_col = np.argmax(dynamic_prog_table[:, n_cols - 1])
    trace_array.append(list(dict_names_trans_mat.keys())[list(dict_names_trans_mat.values()).index(max_idx_last_col)])

    return (dynamic_prog_table, trace_array)


dynamic_prog_table, trace_array = viterbi(["Hung", "Scared", "Tired"])
# print("log M\n", log_m, "\n")
#
# print("log o\n", log_o, "\n")

print("viterbi table dynamic program \n", dynamic_prog_table)

print("find state in sequence observes HungOver, Scared , Tired \n", trace_array)