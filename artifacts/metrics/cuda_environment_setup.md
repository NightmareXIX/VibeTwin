# CUDA Environment Setup Note

## Status

- Date: `2026-04-13`
- Working directory: `.`
- Target Python version: `3.13.7`
- Python launcher used: `py -3.13`
- Environment path: `local CUDA environment`
- GPU seen by `nvidia-smi`: `NVIDIA GeForce RTX 4060`
- NVIDIA driver: `595.97`
- `nvidia-smi` CUDA version: `13.2`

## Official PyTorch Command Chosen

Based on the official PyTorch pages checked on April 13, 2026:

- Get Started Locally: `https://pytorch.org/get-started/locally/`
- Previous Versions: `https://pytorch.org/get-started/previous-versions/`

The exact official Windows pip command selected was:

```powershell
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

The `local CUDA environment` equivalent used in this repo was:

```powershell
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

## What Succeeded

- Created a fresh `local CUDA environment`
- Bootstrapped `pip`, `setuptools`, and `wheel`
- Installed:
  - `pip 26.0.1`
  - `setuptools 82.0.1`
  - `wheel 0.46.3`
  - `numpy 2.4.4`
  - `pillow 12.2.0`

## What Failed

- CUDA PyTorch did **not** install successfully.
- `torch` is not present in `local CUDA environment`.
- `torch.version.cuda` is therefore unavailable.
- `torch.cuda.is_available()` could not be verified inside `local CUDA environment`.

## Exact Issues Encountered

1. Repeated SSL/TLS stream failures while downloading larger packages from both PyPI and the official PyTorch wheel host:

```text
ssl.SSLError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac
```

2. Direct official pip install of CUDA PyTorch failed while downloading:

```text
https://download-r2.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp313-cp313-win_amd64.whl
```

3. The one careful fallback path also failed:

```text
curl.exe ... https://download.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp313-cp313-win_amd64.whl
curl: (56) schannel: failed to read data from server: SEC_E_DECRYPT_FAILURE
```

4. Non-torch dependency installation is still incomplete because `scipy` hit the same SSL/TLS failure during download.

## Current Environment State

- `.venv-ae` still exists and was not modified.
- `local CUDA environment` exists but is incomplete.
- Installed in `local CUDA environment`: `pip`, `setuptools`, `wheel`, `numpy`, `pillow`
- Missing for the planned training run: `torch`, `torchvision`, `torchaudio`, and several non-torch dependencies including `scipy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`, `tqdm`, `pyyaml`, `h5py`, `psutil`, `jupyter`, `ipykernel`

## Manual Files Needed If Network Problems Persist

At minimum, place these official CUDA wheels locally:

- `torch-2.10.0+cu130-cp313-cp313-win_amd64.whl`
- `torchvision-0.25.0+cu130-cp313-cp313-win_amd64.whl`
- `torchaudio-2.10.0+cu130-cp313-cp313-win_amd64.whl`

The current PyPI dependency blocker that still needs a successful download or local wheel is:

- `scipy-1.17.1-cp313-cp313-win_amd64.whl`

## Recommended Next Command After Completing The Environment

Once the missing wheels and dependencies are installed cleanly:

```powershell
python scripts\train_generative_upgrades.py
```
