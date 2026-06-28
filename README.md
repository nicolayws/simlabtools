# SIMLab Tools

Python utilities for experimental mechanics developed at SIMLab, NTNU.

The package aims to provide reusable tools for processing, analysing and visualising experimental data from impact testing, Digital Image Correlation (DIC), radiography and numerical simulations.

---

## Documentation

The latest documentation is available at:

**https://nicolayws.github.io/simlabtools/**

---

## Features

Current functionality includes:

* Reading and synchronising data from the SIMLab Kicking Machine.
* Synchronisation of force and high-speed camera measurements.
* Automatic interpolation of force measurements to camera frames.
* Experimental data post-processing and visualisation.

Planned functionality:

* DIC data processing utilities.
* X-ray and radiography tools.
* Abaqus utilities.
* General plotting utilities.
* Material testing utilities.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nicolayws/simlabtools.git
```

Navigate to the project root:

```bash
cd simlabtools
```

Install the package in editable mode:

```bash
pip install -e .
```

Editable installation ensures that any changes made to the source code are immediately available without reinstalling the package.

---

## Dependencies

Main dependencies:

* NumPy
* SciPy
* Matplotlib

Additional dependencies may be required for specific modules.

---

## Quick Start

```python
from pathlib import Path
from simlabtools import KickReader

fpath = Path(r"D:\Aluforsøk_24.06.2026\Test1.txt")

kick = KickReader(fpath)

kick.sync(
    force_idx=1017,
    camera_idx=617
)

kick.print_summary()
```

Plot force-displacement response:

```python
import matplotlib.pyplot as plt

plt.plot(
    kick.displacement,
    kick.force
)

plt.xlabel("Displacement [mm]")
plt.ylabel("Force [kN]")
plt.show()
```

---

## Example Output

```python
summary = kick.summary()

print(summary)
```

Example:

```text
{
    'name': 'Test1',
    'initial_velocity_m_s': 14.8,
    'max_force_kN': 25.6,
    'max_displacement_mm': 123.4,
    'test_duration_ms': 34.1
}
```

---

## Package Structure

```text
simlabtools/
│
├── kick/          # Kicking machine tools
├── dic/           # Digital Image Correlation tools
├── xray/          # X-ray and radiography tools
├── abaqus/        # Abaqus utilities
├── plotting/      # Plotting utilities
└── utils/         # General helper functions
```

---

## Documentation Development

To preview the documentation locally:

```bash
mkdocs serve
```

The documentation will be available at:

```text
http://127.0.0.1:8000
```

To deploy documentation manually:

```bash
mkdocs gh-deploy
```

Documentation is automatically updated through GitHub Actions whenever changes are pushed to the `main` branch.

---

## Contributing

Contributions, bug reports and feature requests are welcome.

Please open an issue or submit a pull request.

---

## Author

Nicolay Winding-Sørensen

Department of Structural Engineering
Norwegian University of Science and Technology (NTNU)

---

## License

Currently no license has been specified.

Until a license is added, all rights are reserved by the author.
