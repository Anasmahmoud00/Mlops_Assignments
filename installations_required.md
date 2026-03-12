# Assignment 1 - Installations Required

To run Student A's notebook on a local machine, the following setup is required.

## Required Libraries
The project depends on the following Python libraries:
- `numpy`
- `pandas`
- `torch`
- `torchvision`
- `matplotlib`

## Installation Command
Run the following command in your terminal to install all dependencies at once:

```bash
pip install numpy pandas torch torchvision matplotlib
```

## Setup Timeline (Observations)
- **Start Time:** 08:00 PM
- **Installation Command Duration:** ~7 minutes
- **Script Run Duration:** ~7 minutes (including 4 minutes for model training)

## Known Issues
> [!WARNING]
> **Disk Space:** Ensure you have enough disk space (at least 2-3 GB for Torch and its dependencies). If you encounter `OSError: [Errno 28] No space left on device`, you must clear space or expand your storage.

> [!IMPORTANT]
> **Data Path:** The original code references `/kaggle/input/`. Ensure the file `train.csv` is correctly placed in a `./digit-recognizer/` directory relative to the script.
