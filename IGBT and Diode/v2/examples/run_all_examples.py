"""Run every study (each in its own process) and collect all summaries into results/SUMMARY.md."""
import subprocess, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
scripts = sorted(p for p in here.glob("[0-9][0-9]_*.py"))
log = []
if "--collect-only" in sys.argv:
    scripts = []
for s in scripts:
    t0 = time.time()
    rc = subprocess.run([sys.executable, str(s)], cwd=here, capture_output=True, text=True)
    log.append(f"- {s.name}: {'ok' if rc.returncode == 0 else 'FAILED'} ({time.time() - t0:.0f} s)")
    print(log[-1])
    if rc.returncode != 0:
        print(rc.stderr[-2000:])
parts = ["# PEARL v3 example studies - all results", "", *log, ""]
for d in sorted((here / "results").glob("[0-9][0-9]_*")):
    f = d / "summary.md"
    if f.exists():
        parts += [f.read_text().replace("\n# ", "\n## ").replace("# ", "## ", 1), f"Files: results/{d.name}/", ""]
(here / "results" / "SUMMARY.md").write_text("\n".join(parts))
print("written", here / "results" / "SUMMARY.md")
