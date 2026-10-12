"""
Thermal_model.py  (PEARL v3)
============================

The thermal path junction -> case -> paste -> heat sink -> ambient is a LINEAR RC network.
A linear network can be split into independent "modes" (eigenvectors). Each mode behaves
like one resistor + one capacitor with its own time constant tau_m. That gives two tools:

  1. Frequency domain (for the 50 Hz ripple)
         Zth(j*omega) = sum_m  residue_m / (j*omega - lambda_m)        lambda_m = -1/tau_m
     At omega = 0 this is the familiar Rth. At 50 Hz the heat sink modes give ~0.

  2. Time domain at the mission-profile step (for the 1-s mean temperature)
         z_m[k+1] = a_m * z_m[k] + g_m * input_m[k],   a_m = exp(-dt/tau_m)
     This is the EXACT solution for inputs that are constant within each step, so there is
     no ODE solver, no tolerance and no stability limit. Modes much faster than dt simply
     become "P * r" (quasi-static) automatically.

Junction models
  "foster" (default): datasheet Foster Zth_jc used as defined by the datasheet (junction rise
                      above case), T_case from the paste + heat sink chain driven by P_I + P_D.
  "ladder"          : datasheet Foster values inserted into a Cauer chain (PEARL v1). Kept only
                      to reproduce old results; it under-predicts the 50 Hz ripple by ~30 %.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import lfilter

OUTPUTS = ("Tj_IGBT", "Tj_Diode", "T_case", "T_sink")


def _conductance(n, links):
    """Conductance matrix from links (node_a, node_b or None = ambient, R)."""
    G = np.zeros((n, n))
    for a, b, r in links:
        g = 1.0 / r
        G[a, a] += g
        if b is not None:
            G[b, b] += g
            G[a, b] -= g
            G[b, a] -= g
    return G


class ThermalNetwork:
    """
    dz/dt = lam * z + B @ u,   u = [P_IGBT, P_Diode, T_amb]
    y     = W_out @ z,         y = [Tj_IGBT, Tj_Diode, T_case, T_sink]
    """

    def __init__(self, lam, B, W_out, z_uniform, name):
        self.lam = np.asarray(lam, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.W_out = np.asarray(W_out, dtype=np.float64)
        self._z_uniform = z_uniform
        self.name = name

    # ------------------------------------------------------------------ build -----------
    @classmethod
    def from_params(cls, p):
        model = getattr(p, "junction_model", "foster")
        r_paste = np.atleast_1d(p.r_paste).astype(float); cap_paste = np.atleast_1d(p.cap_paste).astype(float)
        r_sink = np.atleast_1d(p.r_sink).astype(float); cap_sink = np.atleast_1d(p.cap_sink).astype(float)

        if model == "foster":
            r_I, tau_I = np.atleast_1d(p.r_I).astype(float), np.atleast_1d(p.tau_I).astype(float)
            r_D, tau_D = np.atleast_1d(p.r_D).astype(float), np.atleast_1d(p.tau_D).astype(float)
            n_p, n_s = len(r_paste), len(r_sink)
            n = n_p + n_s
            links = [(j, j + 1, r_paste[j]) for j in range(n_p)]
            links += [(n_p + j, n_p + j + 1, r_sink[j]) for j in range(n_s - 1)]
            links += [(n - 1, None, r_sink[-1])]
            Cv = np.r_[cap_paste, cap_sink]
            F = np.zeros((n, 3)); F[0, 0] = F[0, 1] = 1.0; F[n - 1, 2] = 1.0 / r_sink[-1]
            mu, W = eigh(_conductance(n, links), np.diag(Cv))       # W.T C W = I
            nI, nD = len(r_I), len(r_D)
            lam = np.r_[-mu, -1.0 / tau_I, -1.0 / tau_D]
            B = np.zeros((n + nI + nD, 3))
            B[:n] = W.T @ F
            B[n:n + nI, 0] = r_I / tau_I            # 1 / C_i of each Foster element
            B[n + nI:, 1] = r_D / tau_D
            W_out = np.zeros((4, n + nI + nD))
            W_out[0, :n] = W[0]; W_out[0, n:n + nI] = 1.0
            W_out[1, :n] = W[0]; W_out[1, n + nI:] = 1.0
            W_out[2, :n] = W[0]
            W_out[3, :n] = W[n_p]

            def z_uniform(T0):
                z = np.zeros(lam.size); z[:n] = W.T @ (Cv * T0); return z
            return cls(lam, B, W_out, z_uniform, "foster")

        if model == "ladder":
            r_I, cap_I = np.atleast_1d(p.r_I).astype(float), np.atleast_1d(p.cap_I).astype(float)
            r_D, cap_D = np.atleast_1d(p.r_D).astype(float), np.atleast_1d(p.cap_D).astype(float)
            n_i, n_d, n_p, n_s = len(r_I), len(r_D), len(r_paste), len(r_sink)
            n = n_i + n_d + n_p + n_s
            iI, iD = np.arange(n_i), n_i + np.arange(n_d)
            iP, iS = n_i + n_d + np.arange(n_p), n_i + n_d + n_p + np.arange(n_s)
            links = [(iI[j], iI[j + 1] if j + 1 < n_i else iP[0], r_I[j]) for j in range(n_i)]
            links += [(iD[j], iD[j + 1] if j + 1 < n_d else iP[0], r_D[j]) for j in range(n_d)]
            links += [(iP[j], iP[j + 1] if j + 1 < n_p else iS[0], r_paste[j]) for j in range(n_p)]
            links += [(iS[j], iS[j + 1], r_sink[j]) for j in range(n_s - 1)] + [(iS[-1], None, r_sink[-1])]
            Cv = np.r_[cap_I, cap_D, cap_paste, cap_sink]
            F = np.zeros((n, 3)); F[iI[0], 0] = 1.0; F[iD[0], 1] = 1.0; F[iS[-1], 2] = 1.0 / r_sink[-1]
            mu, W = eigh(_conductance(n, links), np.diag(Cv))
            net = cls(-mu, W.T @ F, W[[iI[0], iD[0], iP[0], iS[0]], :], lambda T0: W.T @ (Cv * T0), "ladder")
            net._W_full, net._C_full, net._groups = W, Cv, (n_i, n_d, n_p, n_s)
            return net
        raise ValueError("junction_model must be 'foster' or 'ladder'")

    # --------------------------------------------------------------- frequency domain ---
    def H(self, omega):
        """Transfer functions [P_IGBT, P_Diode] -> outputs at angular frequencies omega.
        Shape (4, 2, len(omega)). H(0) is the Rth matrix [K/W]."""
        w = np.atleast_1d(np.asarray(omega, dtype=np.float64))
        return np.einsum("om,mi,mw->oiw", self.W_out, self.B[:, :2], 1.0 / (1j * w[None, :] - self.lam[:, None]))

    @property
    def Rth(self):
        return self.H([0.0])[..., 0].real

    @property
    def time_constants(self):
        return np.sort(-1.0 / self.lam)

    # ------------------------------------------------------------------- states ---------
    def initial_state(self, T0):
        return self._z_uniform(float(T0))

    def steady_state(self, P_I, P_D, T_amb):
        return -(self.B @ np.array([P_I, P_D, T_amb], dtype=np.float64)) / self.lam

    # ------------------------------------------------------------- time domain ----------
    def interval_mean(self, P_I, P_D, T_amb, dt, z0, quasi_static_tau=0.0, block=1_000_000):
        """Exact average of every output over every step [k*dt, (k+1)*dt] for inputs that are
        constant within a step. Modes with tau < quasi_static_tau are treated as instantaneous."""
        N = len(P_I); lam = self.lam
        a = np.exp(lam * dt); g = np.expm1(lam * dt) / lam; h = g / dt
        T_amb = np.broadcast_to(np.asarray(T_amb, dtype=np.float64), (N,))
        Y = np.empty((self.W_out.shape[0], N)); z = np.array(z0, dtype=np.float64, copy=True)
        for s in range(0, N, block):
            e = min(N, s + block)
            Wm = self.B @ np.vstack([P_I[s:e], P_D[s:e], T_amb[s:e]])
            Z = np.empty_like(Wm)
            for m in range(lam.size):
                zss = -Wm[m] / lam[m]
                if -1.0 / lam[m] < quasi_static_tau:
                    Z[m] = zss; z[m] = zss[-1]; continue
                y, _ = lfilter([g[m]], [1.0, -a[m]], Wm[m], zi=[a[m] * z[m]])     # y[k] = z(k+1)
                zk = np.r_[z[m], y[:-1]]
                Z[m] = zss + (zk - zss) * h[m]
                z[m] = y[-1]
            Y[:, s:e] = self.W_out @ Z
        return Y, z

    def simulate(self, P_I, P_D, T_amb, dt, z0, block=1_000_000):
        """Exact response sampled at t_k = k*dt (brute-force reference for short windows)."""
        N = len(P_I); lam = self.lam
        a = np.exp(lam * dt); g = np.expm1(lam * dt) / lam
        T_amb = np.broadcast_to(np.asarray(T_amb, dtype=np.float64), (N,))
        Y = np.empty((self.W_out.shape[0], N)); z = np.array(z0, dtype=np.float64, copy=True)
        for s in range(0, N, block):
            e = min(N, s + block)
            Wm = self.B @ np.vstack([P_I[s:e], P_D[s:e], T_amb[s:e]])
            Z = np.empty_like(Wm)
            for m in range(lam.size):
                y, _ = lfilter([g[m]], [1.0, -a[m]], Wm[m], zi=[a[m] * z[m]])
                Z[m, 0] = z[m]; Z[m, 1:] = y[:-1]; z[m] = y[-1]
            Y[:, s:e] = self.W_out @ Z
        return Y, z


def ripple_from_losses(net, P_I, P_D, f0):
    """
    Periodic steady-state junction ripple (zero mean) for one period of losses per row:
        Tj~(t) = sum_{n>=1} Re{ 2 * P(n) * Zth(j n omega0) * exp(j n omega0 t) }
    Returns (ripple_IGBT, ripple_Diode), each (rows, Np).
    """
    Np = P_I.shape[1]
    w = 2.0 * np.pi * f0 * np.arange(Np // 2 + 1)
    Hw = net.H(w)
    Hw[..., 0] = 0.0                                  # the mean is handled with Rth
    FI, FD = np.fft.rfft(P_I, axis=1), np.fft.rfft(P_D, axis=1)
    R_I = np.fft.irfft(FI * Hw[0, 0] + FD * Hw[0, 1], n=Np, axis=1)
    R_D = np.fft.irfft(FI * Hw[1, 0] + FD * Hw[1, 1], n=Np, axis=1)
    return R_I, R_D


def simulate_igbt_diode_cauer(r_I, cap_I, r_D, cap_D, r_paste, cap_paste, r_sink, cap_sink,
                              P_I, P_D, T_env, dt, method=None, rtol=None, atol=None, debug=False, T0_init=None):
    """PEARL v1 signature, now solved exactly (method/rtol/atol ignored). Returns all ladder nodes."""
    class _P: pass
    p = _P()
    p.junction_model = "ladder"
    p.r_I, p.cap_I, p.r_D, p.cap_D = r_I, cap_I, r_D, cap_D
    p.r_paste, p.cap_paste, p.r_sink, p.cap_sink = r_paste, cap_paste, r_sink, cap_sink
    net = ThermalNetwork.from_params(p)
    T_env = np.asarray(T_env, dtype=np.float64)
    z0 = net.initial_state(T_env[0]) if T0_init is None else net._W_full.T @ (net._C_full * np.asarray(T0_init))
    full = ThermalNetwork(net.lam, net.B, net._W_full, None, "ladder")
    Y, _ = full.simulate(np.asarray(P_I, float), np.asarray(P_D, float), T_env, dt, z0)
    n_i, n_d, n_p, n_s = net._groups
    return (np.arange(Y.shape[1]) * dt, Y[:n_i], Y[n_i:n_i + n_d], Y[n_i + n_d:n_i + n_d + n_p], Y[n_i + n_d + n_p:])
