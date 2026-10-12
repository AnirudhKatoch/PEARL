"""
Lifetime_models.py  (PEARL v3)
==============================

Choose how cycles-to-failure Nf is computed. Every model answers the same question:
    Nf = f(dTj, Tjm, t_on)       (Tj_min = Tjm - dTj/2, Tj_max = Tjm + dTj/2 are derived)
so PEARL can switch models without touching the rest of the code.

Built-in models ("model" key)                           Typical use
  "semikron2024"   Semikron Danfoss AN 21-001 (2024)    default; Semikron-type modules
  "lesit"          Held et al. 1997 (LESIT)             classic reference model
  "bayerer2008"    Bayerer et al., CIPS 2008            soldered modules with Al bond wires
  "cma"            generic Coffin-Manson-Arrhenius      your own A, n, Ea (or fitted from points)
  "curves"         digitised manufacturer curves        best choice when the manufacturer gives curves

Ways to select (Input_parameters.lifetime_model):
  "semikron2024"                                  one built-in model, default parameters
  {"model": "lesit", "A": 640, "alpha": -5}       built-in model, your parameters
  {"model": "curves", ...}                        your curves (see USER_CURVES_EXAMPLE)
  ["semikron2024", "lesit", {...}]                several at once: PEARL reports every one
                                                  (the first one is the main result)

Every model applies its own t_on validity limits (t_on_min / t_on_max), because the
time correction of each model is only fitted over its test range.
"""

import numpy as np

KB = 1.380649e-23          # Boltzmann constant [J/K]
R_GAS = 8.314462618        # gas constant [J/(mol K)]


# ======================================================================================
# Base class
# ======================================================================================

class LifetimeModel:
    """Subclasses implement log_Nf(dT, Tjm, ton, device, pert). pert = dict of relative
    perturbations {"A": e_A, "alpha": e_alpha} (arrays broadcastable to dT) for Monte Carlo."""
    name = "base"
    t_on_min = None
    t_on_max = None

    def clamp_ton(self, ton):
        ton = np.asarray(ton, dtype=np.float64)
        if self.t_on_min is not None:
            ton = np.maximum(ton, self.t_on_min)
        if self.t_on_max is not None:
            ton = np.minimum(ton, self.t_on_max)
        return ton

    def bind(self, device):
        """Callable Nf(dT, Tjm, ton, pert=None) for 'IGBT' or 'Diode'."""
        model = self

        def Nf(dT, Tjm, ton, pert=None):
            dT = np.asarray(dT, dtype=np.float64)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                ln = model.log_Nf(dT, np.asarray(Tjm, dtype=np.float64), model.clamp_ton(ton), device, pert or {})
                out = np.exp(np.clip(ln, -700, 700))
            return np.where(dT > 0, out, np.inf)
        Nf.model_name = model.name
        return Nf

    def describe(self):
        return {"model": self.name}


# ======================================================================================
# Built-in models
# ======================================================================================

class Semikron2024(LifetimeModel):
    """Semikron Danfoss AN 21-001 rev. 02 (2024):
    Nf = A0 * A1^beta * dTj^(alpha-beta) * exp(Ea/(kB*Tjm)) * (C + t_on^gamma)/(C + 2^gamma) * k_thickness,
    beta = exp(-(dTj - T0)/lambda). Parameter sets: 'baseplate' (default), 'no_baseplate', 'sintered'.
    The note states no further increase of Nf for t_on < 40 ms -> t_on_min = 0.04 s."""
    name = "semikron2024"
    SETS = {
        "baseplate":    dict(A0=2.9e9, A1=60, T0_K=40, lambda_K=17, alpha=-4.3, Ea_J=4.50e-20, C=1.0, gamma=-0.75),
        "no_baseplate": dict(A0=2.9e9, A1=60, T0_K=40, lambda_K=17, alpha=-4.3, Ea_J=4.50e-20, C=0.38, gamma=-0.7),
        "sintered":     dict(A0=2.05e11, A1=60, T0_K=38, lambda_K=17, alpha=-4.3, Ea_J=2.54e-20, C=1.44, gamma=-1.21),
    }

    def __init__(self, parameter_set="baseplate", k_thickness=None, t_on_min=0.04, **override):
        self.p = dict(self.SETS[parameter_set]); self.p.update(override)
        self.k = {"IGBT": 1.0, "Diode": 0.5}
        if k_thickness:
            self.k.update(k_thickness)
        self.t_on_min = t_on_min
        self.parameter_set = parameter_set

    def log_Nf(self, dT, Tjm, ton, device, pert):
        p = self.p
        alpha = p["alpha"] * (1.0 + pert.get("alpha", 0.0))
        beta = np.exp(-(dT - p["T0_K"]) / p["lambda_K"])
        return (np.log(p["A0"]) + np.log1p(pert.get("A", 0.0)) + beta * np.log(p["A1"]) + (alpha - beta) * np.log(dT)
                + p["Ea_J"] / (p.get("kB_J_per_K", KB) * Tjm)
                + np.log((p["C"] + ton ** p["gamma"]) / (p["C"] + 2.0 ** p["gamma"]))
                + np.log(self.k[device]))

    def describe(self):
        return {"model": self.name, "parameter_set": self.parameter_set, **self.p, "k_thickness": self.k,
                "t_on_min": self.t_on_min}


class LESIT(LifetimeModel):
    """Held et al. 1997 (LESIT): Nf = A * dTj^alpha * exp(Q/(R*Tjm)), as usually quoted
    A = 640, alpha = -5, Q = 78 kJ/mol. No t_on dependence."""
    name = "lesit"

    def __init__(self, A=640.0, alpha=-5.0, Q_J_per_mol=7.8e4):
        self.A, self.alpha, self.Q = A, alpha, Q_J_per_mol

    def log_Nf(self, dT, Tjm, ton, device, pert):
        alpha = self.alpha * (1.0 + pert.get("alpha", 0.0))
        return np.log(self.A) + np.log1p(pert.get("A", 0.0)) + alpha * np.log(dT) + self.Q / (R_GAS * Tjm)

    def describe(self):
        return {"model": self.name, "A": self.A, "alpha": self.alpha, "Q_J_per_mol": self.Q}


class Bayerer2008(LifetimeModel):
    """Bayerer et al., CIPS 2008:
    Nf = K * dTj^b1 * exp(b2/Tj_min[K]) * t_on^b3 * I^b4 * V^b5 * D^b6
      I = current per bond wire [A], V = blocking voltage / 100 V, D = bond-wire diameter [um].
    t_on limited to the test range (1..15 s) by default. Set I, V, D for YOUR device:
    the defaults are placeholders."""
    name = "bayerer2008"

    def __init__(self, K=9.30e14, b1=-4.416, b2=1285.0, b3=-0.463, b4=-0.716, b5=-0.761, b6=-0.5,
                 I_per_wire_A=10.0, V_blocking_V=600.0, D_wire_um=300.0, t_on_min=1.0, t_on_max=15.0):
        self.c = dict(K=K, b1=b1, b2=b2, b3=b3, b4=b4, b5=b5, b6=b6)
        self.I, self.V, self.D = I_per_wire_A, V_blocking_V / 100.0, D_wire_um
        self.t_on_min, self.t_on_max = t_on_min, t_on_max

    def log_Nf(self, dT, Tjm, ton, device, pert):
        c = self.c
        b1 = c["b1"] * (1.0 + pert.get("alpha", 0.0))
        Tmin = Tjm - 0.5 * dT
        return (np.log(c["K"]) + np.log1p(pert.get("A", 0.0)) + b1 * np.log(dT) + c["b2"] / Tmin
                + c["b3"] * np.log(ton) + c["b4"] * np.log(self.I) + c["b5"] * np.log(self.V) + c["b6"] * np.log(self.D))

    def describe(self):
        return {"model": self.name, **self.c, "I_per_wire_A": self.I, "V_blocking_V": self.V * 100,
                "D_wire_um": self.D, "t_on_min": self.t_on_min, "t_on_max": self.t_on_max}


class CoffinMansonArrhenius(LifetimeModel):
    """Generic: Nf = A * dTj^(-n) * exp(Ea/(kB*T)) * time factor
    T = 'Tjm' | 'Tj_max' | 'Tj_min' (which temperature your data refers to).
    Optional time factor (C + t_on^gamma)/(C + t_ref^gamma) if C and gamma are given."""
    name = "cma"

    def __init__(self, A, n, Ea_J=0.0, temperature="Tjm", C=None, gamma=None, t_ref=2.0, t_on_min=None,
                 t_on_max=None, per_device=None):
        self.A, self.n, self.Ea, self.temperature = A, n, Ea_J, temperature
        self.C, self.gamma, self.t_ref = C, gamma, t_ref
        self.t_on_min, self.t_on_max = t_on_min, t_on_max
        self.per_device = per_device or {}          # e.g. {"Diode": {"A": ...}} to override per device

    def log_Nf(self, dT, Tjm, ton, device, pert):
        A = self.per_device.get(device, {}).get("A", self.A)
        n = self.per_device.get(device, {}).get("n", self.n) * (1.0 + pert.get("alpha", 0.0))
        T = {"Tjm": Tjm, "Tj_max": Tjm + 0.5 * dT, "Tj_min": Tjm - 0.5 * dT}[self.temperature]
        ln = np.log(A) + np.log1p(pert.get("A", 0.0)) - n * np.log(dT) + self.Ea / (KB * T)
        if self.C is not None and self.gamma is not None:
            ln = ln + np.log((self.C + ton ** self.gamma) / (self.C + self.t_ref ** self.gamma))
        return ln

    def describe(self):
        return {"model": self.name, "A": self.A, "n": self.n, "Ea_J": self.Ea, "temperature": self.temperature,
                "C": self.C, "gamma": self.gamma, "t_ref": self.t_ref, "per_device": self.per_device}


# ======================================================================================
# User-defined curves
# ======================================================================================

USER_CURVES_EXAMPLE = {
    "model": "curves",
    "name": "my_module_PC_curves",
    # which junction temperature labels each curve: "Tj_max" (most datasheet/AN curves), "Tjm" or "Tj_min"
    "temperature": "Tj_max",
    # digitised curves, keyed by that temperature in degC; dTj in K, Nf in cycles.
    # Give "IGBT" and "Diode" separately, or only "IGBT" and set "Diode": "IGBT" to reuse it.
    "curves": {
        "IGBT": {
            125: {"dTj": [30, 40, 60, 80, 100], "Nf": [6.0e7, 1.2e7, 1.4e6, 3.0e5, 9.0e4]},
            150: {"dTj": [30, 40, 60, 80, 100], "Nf": [3.0e7, 6.0e6, 7.0e5, 1.5e5, 4.5e4]},
        },
        "Diode": "IGBT",
    },
    "t_on_s": 2.0,                                      # t_on at which the curves were measured
    "ton_correction": {"C": 1.0, "gamma": -0.75},       # how Nf scales with t_on (None = no scaling)
    "t_on_min": 0.04,                                    # lower limit of the time correction
    "Ea_J": None,                                       # only used if a single temperature curve is given
    "extrapolate_dT": "power",                          # outside the dTj range: "power" (end slope) | "clamp"
    "extrapolate_T": "arrhenius",                       # outside the temperature range: "arrhenius" | "clamp"
    "failure_probability": 0.05,                        # what the curves represent (stored for the report)
}


class UserCurves(LifetimeModel):
    """Nf from digitised curves: log-log interpolation in dTj along each curve, linear in 1/T
    (Arrhenius-consistent) between curves of different temperature."""
    name = "curves"

    def __init__(self, spec):
        self.spec = dict(spec)
        self.name = spec.get("name", "curves")
        self.temperature = spec.get("temperature", "Tj_max")
        self.t_on_s = spec.get("t_on_s", None)
        self.tc = spec.get("ton_correction", None)
        self.t_on_min = spec.get("t_on_min", None)
        self.t_on_max = spec.get("t_on_max", None)
        self.Ea = spec.get("Ea_J", None)
        self.ext_dT = spec.get("extrapolate_dT", "power")
        self.ext_T = spec.get("extrapolate_T", "arrhenius")
        curves = spec["curves"]
        if "IGBT" not in curves:                       # one set for both devices
            curves = {"IGBT": curves, "Diode": "IGBT"}
        self.tables = {}
        for dev in ("IGBT", "Diode"):
            c = curves.get(dev, "IGBT")
            c = curves[c] if isinstance(c, str) else c
            temps = sorted(float(t) for t in c)
            rows = []
            for t in temps:
                key = t if t in c else (int(t) if int(t) in c else str(t))
                x = np.log(np.asarray(c[key]["dTj"], dtype=np.float64))
                y = np.log(np.asarray(c[key]["Nf"], dtype=np.float64))
                o = np.argsort(x)
                rows.append((x[o], y[o]))
            self.tables[dev] = (np.array(temps) + 273.15, rows)

    def _logNf_curve(self, x, y, lx):
        out = np.interp(lx, x, y)
        if self.ext_dT == "power" and x.size >= 2:
            lo, hi = lx < x[0], lx > x[-1]
            s0 = (y[1] - y[0]) / (x[1] - x[0]); s1 = (y[-1] - y[-2]) / (x[-1] - x[-2])
            out = np.where(lo, y[0] + s0 * (lx - x[0]), out)
            out = np.where(hi, y[-1] + s1 * (lx - x[-1]), out)
        return out

    def log_Nf(self, dT, Tjm, ton, device, pert):
        T_curve, rows = self.tables[device]
        T = {"Tjm": Tjm, "Tj_max": Tjm + 0.5 * dT, "Tj_min": Tjm - 0.5 * dT}[self.temperature]
        lx = np.log(dT)
        L = np.stack([self._logNf_curve(x, y, lx) for x, y in rows])        # (n_curves, ...)
        if len(rows) == 1:
            ln = L[0]
            if self.Ea:
                ln = ln + self.Ea / KB * (1.0 / T - 1.0 / T_curve[0])
        else:
            u = 1.0 / T; uc = 1.0 / T_curve                                   # uc decreasing with T
            order = np.argsort(uc); uc = uc[order]; L = L[order]
            if self.ext_T == "clamp":
                u = np.clip(u, uc[0], uc[-1])
            j = np.clip(np.searchsorted(uc, u) - 1, 0, uc.size - 2)
            u0, u1 = uc[j], uc[j + 1]
            L0 = np.take_along_axis(L, j[None, ...], 0)[0] if L.ndim > 1 and np.ndim(j) else L[j]
            L1 = np.take_along_axis(L, (j + 1)[None, ...], 0)[0] if L.ndim > 1 and np.ndim(j) else L[j + 1]
            ln = L0 + (L1 - L0) * (u - u0) / (u1 - u0)
        if self.tc is not None and self.t_on_s is not None:
            C, g = self.tc["C"], self.tc["gamma"]
            ln = ln + np.log((C + ton ** g) / (C + self.t_on_s ** g))
        if "alpha" in pert:                     # Monte Carlo: scale the slope around 50 K
            ln_ref = self.log_Nf(np.full(np.shape(dT), 50.0), Tjm, ton, device, {})
            ln = ln_ref + (ln - ln_ref) * (1.0 + pert["alpha"])
        return ln + np.log1p(pert.get("A", 0.0))

    def describe(self):
        return {k: v for k, v in self.spec.items()}


# ======================================================================================
# Fitting a model to digitised points
# ======================================================================================

def fit_cma(points, temperature="Tjm", fit_Ea=True, t_on=None, C=None, gamma=None):
    """
    Least-squares fit of Nf = A * dTj^(-n) * exp(Ea/(kB*T)) to points
    [(dTj [K], T [degC], Nf), ...] (T = the temperature your data refers to).
    Returns a dict usable as Input_parameters.lifetime_model.
    """
    P = np.asarray(points, dtype=np.float64)
    dT, T, Nf = P[:, 0], P[:, 1] + 273.15, P[:, 2]
    cols = [np.ones_like(dT), -np.log(dT)] + ([1000.0 / T] if fit_Ea else [])     # well-scaled columns
    X = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(X, np.log(Nf), rcond=None)
    resid = np.log(Nf) - X @ coef
    spec = {"model": "cma", "A": float(np.exp(coef[0])), "n": float(coef[1]),
            "Ea_J": float(coef[2] * 1000.0 * KB) if fit_Ea else 0.0, "temperature": temperature,
            "fit_rms_log_error": float(np.sqrt(np.mean(resid ** 2)))}
    if C is not None and gamma is not None:
        spec.update({"C": C, "gamma": gamma, "t_ref": t_on if t_on is not None else 2.0})
    return spec


# ======================================================================================
# Factory
# ======================================================================================

def build_lifetime_model(spec, params=None):
    """spec: name (str), dict with "model", or an already built LifetimeModel.
    params: Input_parameters instance (supplies the Semikron parameters / k_thickness)."""
    if isinstance(spec, LifetimeModel):
        return spec
    if isinstance(spec, str):
        spec = {"model": spec}
    spec = dict(spec)
    label = spec.pop("name", None) if spec.get("model") != "curves" else None
    model = _build(spec, params)
    if label:
        model.name = label
    return model


def _build(spec, params):
    kind = spec.pop("model")
    if kind == "semikron2024" and "parameter_set" in spec and spec["parameter_set"] != "baseplate":
        m = Semikron2024(**spec)
        m.name = f"semikron2024_{spec['parameter_set']}"
        return m
    if kind == "semikron2024":
        if params is not None and "parameter_set" not in spec:
            base = dict(A0=params.A0, A1=params.A1, T0_K=params.T0_K, lambda_K=params.lambda_K, alpha=params.alpha,
                        Ea_J=params.Ea_J, C=params.C, gamma=params.gamma, kB_J_per_K=params.kB_J_per_K)
            base.update({k: v for k, v in spec.items() if k in base})
            floor = params.ton_min if getattr(params, "ton_rule", "semikron") == "semikron" else None
            return Semikron2024("baseplate", k_thickness=spec.get("k_thickness", params.k_thickness),
                                t_on_min=spec.get("t_on_min", floor), **base)
        return Semikron2024(**spec)
    if kind == "lesit":
        return LESIT(**spec)
    if kind == "bayerer2008":
        return Bayerer2008(**spec)
    if kind == "cma":
        spec.pop("fit_rms_log_error", None)
        return CoffinMansonArrhenius(**spec)
    if kind == "curves":
        spec["model"] = "curves"
        return UserCurves(spec)
    raise ValueError(f"Unknown lifetime model '{kind}'")


def build_lifetime_models(spec, params=None):
    """Always returns a list of models (one entry per model to evaluate)."""
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    return [build_lifetime_model(s, params) for s in specs]
