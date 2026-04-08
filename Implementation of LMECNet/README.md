# LMECNet Core Modules

Official PyTorch release of the **core method modules** of LMECNet for single-shot LFOM complex-field reconstruction.

## Overview

This repository releases the main method components of LMECNet rather than the full end-to-end model and training pipeline. The current release includes:

- **SMCCU** for stage-wise spatial-domain state-space modeling  
- **BFMFB** for bottleneck frequency-domain bidirectional state-space modeling  
- **Adaptive gating unit** for directional feature fusion  
- **TAD loss** for explicit discrimination against the conjugate twin-artifact hypothesis  

## Repository Structure

```text
.
├── modules/
│   ├── __init__.py
│   ├── gating.py
│   ├── sequence_utils.py
│   ├── spectral_utils.py
│   ├── smccu.py
│   └── bfmfb.py
├── losses/
│   ├── __init__.py
│   └── tad_loss.py
├── examples/
│   ├── inputs/
│   ├── targets/
│   └── outputs/
└── README.md