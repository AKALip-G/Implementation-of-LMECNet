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
```

## Main Components

- `modules/gating.py`
  Adaptive gating unit used for directional feature fusion.
- `modules/sequence_utils.py`
  Utilities for converting 2D feature maps into row-wise and column-wise token sequences.
- `modules/spectral_utils.py`
  Utilities for FFT/iFFT, paired real/imag channel conversion, radial band masking, and spectral reassembly.
- `modules/smccu.py`
  Implementation of the **Sequential Mamba-CNN Coupling Unit (SMCCU)**.
- `modules/bfmfb.py`
  Implementation of the **Bidirectional Frequency-Enhanced Mamba Fusion Block (BFMFB)**.
- `losses/tad_loss.py`
  Implementation of the finalized **Twin-Artifact Discrimination (TAD) loss**.
- `examples/inputs/`, `examples/targets/`, `examples/outputs/`
  Placeholder directories for example inputs, reference targets, and reconstructed outputs.

## Notes

- This repository focuses on the **core modules** of the method.
