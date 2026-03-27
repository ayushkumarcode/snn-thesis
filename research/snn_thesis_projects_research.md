# SNN Thesis & Student Projects on GitHub

Went through GitHub looking for undergraduate and masters thesis/dissertation projects related to SNNs and neuromorphic computing. Found 40+ projects.

The pattern that emerges is pretty clear: most BSc/3rd-year projects focus on a single well-defined task (usually MNIST or gesture classification), use one established framework (snnTorch, Brian2, SpikingJelly, or BindsNET), compare SNN performance against a conventional ANN baseline, and work with 1-2 datasets. Masters-level projects tend to be more ambitious -- hardware deployment (SpiNNaker, Loihi, FPGA), multi-domain applications (robotics, RL), or novel architectural stuff. Not many projects include formal thesis PDFs in the repo, but several reference external documents.

---

## Confirmed Undergraduate / BSc / Final Year Projects

### Shape Detector SNN (University of Manchester)
- https://github.com/filippoferrari/shape_detector_snn
- Dissertation repo: https://github.com/filippoferrari/bsc_dissertation
- Shape detection using SNNs, BSc AI dissertation supervised by **Steve Furber** (creator of SpiNNaker)
- Uses Python, pyDVS library, custom shape images through DVS simulation
- Results not quantified in README. LaTeX source in dissertation repo.
- Well-structured with tests, CI, configs. 2 stars, 107 commits, archived 2024.
- Good example of a BSc project at a top university. Clean code, proper testing, supervised by a world expert.

### Musical Pattern Recognition in SNNs (BEng Final Year)
- https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks
- SNN that differentiates musical notes from audio sequences
- Brian 2, Brian 1 Hears bridge, NumPy, matplotlib, Mingus music library
- Custom monophonic audio (.wav). Successfully differentiated notes with visualizations of spike patterns, membrane potentials, synaptic weight evolution.
- **Thesis PDF available:** http://amid.fish/beng_project_report.pdf
- 49 stars, 17 forks. That's a lot for a student project -- clearly struck a chord (pun intended). Novel application domain, strong documentation. Good model for an ambitious but achievable BEng project.

### SNN for Digit Recognition (King's College London BSc)
