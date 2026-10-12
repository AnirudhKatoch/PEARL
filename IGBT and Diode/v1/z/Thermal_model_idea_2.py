from numba import njit
import numpy as np



@njit
def rollout_cauer_numba(r_I, cap_I,
                        r_D, cap_D,
                        r_paste, cap_paste,
                        r_sink, cap_sink,
                        P_I, P_D,
                        T_env, dt):

    n_steps = len(P_I)
    mission_duration_s = n_steps * dt

    n_i = len(r_I)
    n_d = len(r_D)
    n_p = len(r_paste)   # should be 1
    n_s = len(r_sink)  # should be 1

    # --- indexing in state vector T ---
    idx_i_start = 0
    idx_d_start = n_i
    idx_p_start = n_i + n_d
    idx_s_start = n_i + n_d + n_p
    total_nodes = n_i + n_d + n_p + n_s

    # initial condition
    T0_amb = T_env[0]
    T = np.full(total_nodes, T0_amb, dtype=np.float64)

    # output: all node temperatures over time
    T_all = np.empty((total_nodes, n_steps), dtype=np.float64)
    time_steps = np.empty(n_steps, dtype=np.float64)



    for k in range(n_steps):
        t = k * dt
        time_steps[k] = t

        P_i = P_I[k]
        P_d = P_D[k]
        Tamb = T_env[k]

        dTdt = np.zeros_like(T)

        # views
        T_i = T[idx_i_start:idx_i_start + n_i]
        T_d = T[idx_d_start:idx_d_start + n_d]
        T_p = T[idx_p_start:idx_p_start + n_p]  # case node(s)
        T_s = T[idx_s_start:idx_s_start + n_s]  # sink node(s)

        # --- IGBT ladder ---
        # junction node
        if n_i == 1:
            # single RC from junction to case
            Tj = T_i[0]
            T_case = T_p[0]
            q_to_case = (Tj - T_case) / r_I[0]
            dTdt[idx_i_start + 0] = (P_i - q_to_case) / cap_I[0]

        else:

            q_i_0_to_1 = (T_i[0] - T_i[1]) / r_I[0]
            dTdt[idx_i_start + 0] = (P_i - q_i_0_to_1) / cap_I[0]

            # internal nodes
            for j in range(1, n_i - 1):
                q_from_prev = (T_i[j - 1] - T_i[j]) / r_I[j - 1]
                q_to_next = (T_i[j] - (T_p[0] if j == n_i - 1 else T_i[j + 1])) / r_I[j]
                dTdt[idx_i_start + j] = (q_from_prev - q_to_next) / cap_I[j]

            # last IGBT node -> case
            if n_i > 1:
                j = n_i - 1
                q_from_prev = (T_i[j - 1] - T_i[j]) / r_I[j - 1]
                q_to_case = (T_i[j] - T_p[0]) / r_I[j]
                dTdt[idx_i_start + j] = (q_from_prev - q_to_case) / cap_I[j]

        # --- Diode ladder ---
        # junction node
        if n_d == 1:
            # single RC from junction directly to case
            Tj_d = T_d[0]  # diode junction temperature
            T_case = T_p[0]
            q_to_case = (Tj_d - T_case) / r_D[0]
            dTdt[idx_d_start + 0] = (P_d - q_to_case) / cap_D[0]

        else:

            q_d_0_to_1 = (T_d[0] - T_d[1]) / r_D[0]
            dTdt[idx_d_start + 0] = (P_d - q_d_0_to_1) / cap_D[0]

            # internal nodes
            for j in range(1, n_d - 1):
                q_from_prev = (T_d[j - 1] - T_d[j]) / r_D[j - 1]
                q_to_next = (T_d[j] - (T_p[0] if j == n_d - 1 else T_d[j + 1])) / r_D[j]
                dTdt[idx_d_start + j] = (q_from_prev - q_to_next) / cap_D[j]

            # last Diode node -> case
            if n_d > 1:
                j = n_d - 1
                q_from_prev = (T_d[j - 1] - T_d[j]) / r_D[j - 1]
                q_to_case = (T_d[j] - T_p[0]) / r_D[j]
                dTdt[idx_d_start + j] = (q_from_prev - q_to_case) / cap_D[j]

        # --- Case node (package) ---
        T_i_last = T_i[-1]
        T_d_last = T_d[-1]
        T_case = T_p[0]
        T_sink = T_s[0]

        q_i_to_case = (T_i_last - T_case) / r_I[-1]
        q_d_to_case = (T_d_last - T_case) / r_D[-1]

        # --- Paste / Case ladder ---

        if n_p == 1:
            # Single case/paste node directly connected to first sink node
            T_case = T_p[0]
            T_sink0 = T_s[0]

            q_case_to_next = (T_case - T_sink0) / r_paste[0]
            dTdt[idx_p_start + 0] = (q_i_to_case + q_d_to_case - q_case_to_next) / cap_paste[0]

        else:
            # j = 0: Case node -> Paste node 1
            T_case = T_p[0]
            q_case_to_next = (T_case - T_p[1]) / r_paste[0]
            dTdt[idx_p_start + 0] = (q_i_to_case + q_d_to_case - q_case_to_next) / cap_paste[0]

            # internal paste nodes 1 .. n_p-2
            for j in range(1, n_p - 1):
                T_j = T_p[j]
                T_jm = T_p[j - 1]  # previous
                T_jp = T_p[j + 1]  # next

                q_from_prev = (T_jm - T_j) / r_paste[j - 1]
                q_to_next = (T_j - T_jp) / r_paste[j]
                dTdt[idx_p_start + j] = (q_from_prev - q_to_next) / cap_paste[j]

            # last paste node j = n_p - 1 -> first sink node T_s[0]
            j = n_p - 1
            T_last_p = T_p[j]
            T_prev_p = T_p[j - 1]
            T_sink0 = T_s[0]

            q_from_prev = (T_prev_p - T_last_p) / r_paste[j - 1]
            q_to_sink0 = (T_last_p - T_sink0) / r_paste[j]
            dTdt[idx_p_start + j] = (q_from_prev - q_to_sink0) / cap_paste[j]

        # --- Sink ladder ---

        # Heat entering the sink ladder from paste side:
        if n_p == 1:
            # when n_p == 1, paste/case connects directly to sink[0]
            q_from_paste = q_case_to_next
        else:
            # when n_p > 1, last paste node connects to sink[0]
            q_from_paste = q_to_sink0

        if n_s == 1:
            # single sink node connected to ambient
            T_sink = T_s[0]

            q_sink_to_amb = (T_sink - Tamb) / r_sink[0]
            dTdt[idx_s_start + 0] = (q_from_paste - q_sink_to_amb) / cap_sink[0]

        else:
            # first sink node: gets heat from paste, sends to sink[1]
            T_s0 = T_s[0]
            T_s1 = T_s[1]
            q_0_to_1 = (T_s0 - T_s1) / r_sink[0]
            dTdt[idx_s_start + 0] = (q_from_paste - q_0_to_1) / cap_sink[0]

            # internal sink nodes: 1 .. n_s-2
            for j in range(1, n_s - 1):
                T_j = T_s[j]
                T_jm = T_s[j - 1]
                T_jp = T_s[j + 1]

                q_from_prev = (T_jm - T_j) / r_sink[j - 1]
                q_to_next = (T_j - T_jp) / r_sink[j]
                dTdt[idx_s_start + j] = (q_from_prev - q_to_next) / cap_sink[j]

            # last sink node j = n_s - 1 -> ambient
            j = n_s - 1
            T_last_s = T_s[j]
            T_prev_s = T_s[j - 1]

            q_from_prev = (T_prev_s - T_last_s) / r_sink[j - 1]
            q_to_amb = (T_last_s - Tamb) / r_sink[j]
            dTdt[idx_s_start + j] = (q_from_prev - q_to_amb) / cap_sink[j]

        # update temperatures (explicit Euler)
        T = T + dt * dTdt
        T_all[:, k] = T

    return T_all, time_steps


def simulate_igbt_diode_cauer(r_I, cap_I,r_D, cap_D,r_paste, cap_paste,r_sink, cap_sink,P_I, P_D,T_env,dt):

    P_I = np.asarray(P_I, dtype=np.float64)
    P_D = np.asarray(P_D, dtype=np.float64)
    T_env = np.asarray(T_env, dtype=np.float64)

    r_I = np.asarray(r_I, dtype=np.float64)
    cap_I = np.asarray(cap_I, dtype=np.float64)
    r_D = np.asarray(r_D, dtype=np.float64)
    cap_D = np.asarray(cap_D, dtype=np.float64)
    r_paste = np.asarray(r_paste, dtype=np.float64)
    cap_paste = np.asarray(cap_paste, dtype=np.float64)
    r_sink = np.asarray(r_sink, dtype=np.float64)
    cap_sink = np.asarray(cap_sink, dtype=np.float64)

    T_all, time_steps = rollout_cauer_numba(r_I, cap_I,r_D, cap_D,r_paste, cap_paste,r_sink, cap_sink,P_I, P_D,T_env,dt)

    n_i = len(r_I)
    n_d = len(r_D)
    n_p = len(r_paste)
    n_s = len(r_sink)

    idx_i_start = 0
    idx_d_start = n_i
    idx_p_start = n_i + n_d
    idx_s_start = n_i + n_d + n_p

    T_i = T_all[idx_i_start:idx_i_start + n_i, :]
    T_d = T_all[idx_d_start:idx_d_start + n_d, :]
    T_p = T_all[idx_p_start:idx_p_start + n_p, :]
    T_s = T_all[idx_s_start:idx_s_start + n_s, :]

    Tj_igbt = T_i[0, :]
    Tj_diode = T_d[0, :]
    T_case = T_p[0, :]
    T_sink = T_s[0, :]

    return time_steps, Tj_igbt, Tj_diode, T_case, T_sink