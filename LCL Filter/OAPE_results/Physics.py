"""
Bounding calculation: how much winding loss is missed by using Rdc for the
LCL filter inductors, given the 0.500 mm Elektrisola strand at fo = 50 Hz,
fsw = 10 kHz (mf = 200).

Two contributions:
  1. Skin effect inside one isolated round strand  -> exact Bessel solution
  2. Proximity effect between layers               -> Dowell (equivalent foil)
"""
import numpy as np
from scipy.special import ber, bei, berp, beip

rho   = 1.709e-8      # ohm.m  copper @20C  (Input_parameters.py)
mu0   = 4*np.pi*1e-7
d     = 0.500e-3      # m      strand diameter (Input_parameters.py)
fo    = 50.0

def delta(f):
    """Skin depth [m]."""
    return np.sqrt(rho/(np.pi*f*mu0))

def FR_skin_round(f):
    """Exact Rac/Rdc for an isolated round conductor (Kelvin/Bessel)."""
    q = d/(np.sqrt(2)*delta(f))          # standard argument
    if q < 1e-6:
        return 1.0
    num = ber(q)*beip(q) - bei(q)*berp(q)
    den = berp(q)**2 + beip(q)**2
    return (q/2)*num/den

def FR_dowell(f, m_layers, eta=0.8):
    """
    Dowell Rac/Rdc for m layers, round wire mapped to equivalent foil.
    eta = porosity factor (fraction of layer width occupied by copper).
    """
    h_eq = d*np.sqrt(np.pi/4)            # round -> square equivalent height
    xi   = (h_eq/delta(f))*np.sqrt(eta)
    if xi < 1e-9:
        return 1.0
    A = (np.sinh(2*xi) + np.sin(2*xi))/(np.cosh(2*xi) - np.cos(2*xi))
    B = (np.sinh(xi)   - np.sin(xi))  /(np.cosh(xi)   + np.cos(xi))
    return xi*(A + (2*(m_layers**2 - 1)/3)*B)

print(f"strand diameter d      = {d*1e3:.3f} mm")
for h in (1, 50, 180, 200, 400):
    f = h*fo
    print(f"  h={h:4d}  f={f/1e3:7.2f} kHz   delta={delta(f)*1e3:6.3f} mm   "
          f"d/delta={d/delta(f):5.2f}   FR_skin={FR_skin_round(f):6.3f}")

print("\nDowell FR (skin + proximity) vs layer count:")
print(f"{'layers':>7}" + "".join(f"{f'h={h}':>12}" for h in (200, 400)))
for m in (1, 2, 4, 6, 8):
    row = "".join(f"{FR_dowell(h*fo, m):12.2f}" for h in (200, 400))
    print(f"{m:>7}" + row)

# ---- harmonic share of winding loss -------------------------------------
# Paper: L1 harmonic ~4.12% of fundamental at both ends of the drift.
#        L2 harmonic rises 0.83% -> 1.19%.
# Sidebands sit at h=200 and h=400; split the harmonic RMS between them.
print("\nHarmonic share of winding loss,  P_w,h / P_w,total")
print("(all harmonic content placed at h=200 and h=400, 70/30 split by RMS)")
print(f"{'element':>12}{'Ih/I1':>9}{'layers':>8}{'share':>10}")
for name, ratio in (("L1", 0.0412), ("L2 nominal", 0.0083), ("L2 drifted", 0.0119)):
    for m in (2, 4, 8):
        s200, s400 = 0.70*ratio, 0.30*ratio      # RMS split
        Ph = FR_dowell(200*fo, m)*s200**2 + FR_dowell(400*fo, m)*s400**2
        P1 = FR_dowell(fo, m)*1.0**2
        print(f"{name:>12}{ratio*100:8.2f}%{m:8d}{100*Ph/(P1+Ph):9.3f}%")