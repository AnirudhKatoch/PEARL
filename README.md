# ⚡ PEARL

PEARL is an open-source Python framework for electro–thermal lifetime assessment of power-electronic converters.
It integrates electrical loss calculation, dynamic thermal RC-network simulation, and physics-based fatigue modeling 
into a unified workflow, enabling accurate reliability estimation under realistic mission profiles. Designed for
research and system-level analysis, PEARL captures the coupled behaviour of electrical loading, thermal cycling, and 
degradation mechanisms that govern wear-out in modern power-converter hardware. The framework provides a reproducible, 
extensible environment for studying lifetime impacts of grid services, reactive-power operation, control strategies,
and device-level design decisions, helping bridge the gap between practical converter operation and rigorous 
reliability evaluation.

---

## 👤 Author
**Anirudh Katoch**  
PhD Candidate, 
Munich Institute of Integrated Materials, Energy and Process Engineering (MEP), 
Technical University of Munich
Lichtenbergstr. 4a 85748 Garching
Supervisor: **Prof. Dr. Thomas Hamacher**  
📧 anirudh.katoch@tum.de  

---


## ⚙️ Execution and Usage

### Requirements
- **Python 3.13+**
- Dependencies listed in `requirements.txt`

---

## 📚 How to Cite

If you use PEARL in your research, please cite the paper(s) describing the module(s) you used.

**IGBT–diode module**  
Katoch, A., Pant, P., & Hamacher, T. (2026). *An Open-Source Tool for Lifetime Estimation of IGBT–Diode
Power Modules: A Module in PEARL.* 2026 10th International Conference on Power Energy Systems and
Applications (ICoPESA 2026), Munich, Germany. Preprint, Zenodo. https://doi.org/10.5281/zenodo.20702777

**DC-link capacitor module**  
Katoch, A., Hamacher, T., & Pant, P. (2026). *An Open-Source Tool for Lifetime Assessment of DC-Link
Capacitors in Grid-Connected Converters.* IEEE PES ISGT Europe 2026, Budapest, Hungary.
Preprint, Zenodo. https://doi.org/10.5281/zenodo.22672994

**LCL filter module**  
Citation will be added upon publication.

### BibTeX

```bibtex
@misc{Katoch2026IGBT,
  author       = {Katoch, Anirudh and Pant, Prashant and Hamacher, Thomas},
  title        = {An Open-Source Tool for Lifetime Estimation of {IGBT}--Diode Power Modules: A Module in {PEARL}},
  year         = {2026},
  howpublished = {Preprint, Zenodo. 2026 10th International Conference on Power Energy Systems
                  and Applications (ICoPESA 2026), Munich, Germany},
  doi          = {10.5281/zenodo.20702777}
}

@misc{Katoch2026DCLink,
  author       = {Katoch, Anirudh and Hamacher, Thomas and Pant, Prashant},
  title        = {An Open-Source Tool for Lifetime Assessment of {DC}-Link Capacitors in Grid-Connected Converters},
  year         = {2026},
  howpublished = {Preprint, Zenodo. IEEE PES ISGT Europe 2026, Budapest, Hungary},
  doi          = {10.5281/zenodo.22672994}
}
```

> **Version DOIs vs. all-versions DOIs:** the DOIs above point to the exact version cited.
> To always link to the latest version, use the all-versions (concept) DOIs:
> IGBT–diode `10.5281/zenodo.20702776`, DC-link `10.5281/zenodo.20703095`.
