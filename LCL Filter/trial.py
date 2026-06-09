import numpy as np

# angles you want to sweep (degrees)
angles = [0, 30, 60]

print(f"{'Angle':>6} | {'PF=cos':>10} | {'pf for LAG':>12} | {'pf for LEAD':>12}")
print("-" * 50)
for a in angles:
    pf = np.cos(np.deg2rad(a))
    if a == 0:
        print(f"{a:>6} | {pf:>10.6f} | {'unity':>12} | {'unity':>12}")
    else:
        print(f"{a:>6} | {pf:>10.6f} | {-pf:>12.6f} | {pf:>12.6f}")