GITHUB NOTEBOOK PAGES
—————-
1. Repository Structure:

```
echo-detection-system/
├── .github/
│   ├── workflows/
│   │   ├── run-simulation.yml     # Automated simulation runs
│   │   ├── run-tests.yml          # Test suite automation
│   │   └── deploy-notebooks.yml   # Deploy notebooks to Pages
├── notebooks/
│   ├── 01_simulation_basics.ipynb
│   ├── 02_real_data_analysis.ipynb
│   ├── 03_parameter_space.ipynb
│   └── 04_merlin_tests.ipynb
├── src/
│   ├── echo_detection_system.py   # Our combined code
│   ├── requirements.txt
│   └── setup.py
├── configs/
│   ├── simulation_config.yaml
│   └── detector_config.yaml
├── outputs/                      # Simulation results
├── docs/
│   └── index.md
├── README.md
└── .gitignore
```

2. GitHub Actions Workflow (run-simulation.yml):

```yaml
name: Run Automated Simulations

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:  # Manual trigger
    inputs:
      simulation_type:
        description: 'Simulation type'
        required: true
        default: 'full'

jobs:
  run-simulation:
    runs-on: ubuntu-latest
    container: python:3.9-slim
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install system dependencies
      run: |
        apt-get update && apt-get install -y \
          gcc \
          g++ \
          libhdf5-dev \
          libopenblas-dev \
          && rm -rf /var/lib/apt/lists/*
    
    - name: Install Python dependencies
      run: |
        pip install --upgrade pip
        pip install -r src/requirements.txt
        pip install jupyter nbconvert papermill
    
    - name: Run simulation notebooks
      run: |
        mkdir -p outputs
        python -m papermill notebooks/01_simulation_basics.ipynb \
          outputs/01_results.ipynb \
          -p total_mass 30 \
          -p detector 'aLIGO' \
          -p n_templates 100
        
        python -m papermill notebooks/02_real_data_analysis.ipynb \
          outputs/02_results.ipynb \
          -p gps_time 1126259462.4 \
          -p detector 'H1'
    
    - name: Convert notebooks to HTML
      run: |
        jupyter nbconvert --to html outputs/*.ipynb
        jupyter nbconvert --to pdf outputs/*.ipynb
    
    - name: Upload results as artifact
      uses: actions/upload-artifact@v3
      with:
        name: simulation-results
        path: |
          outputs/
          notebooks/
        retention-days: 30
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: $%20secrets.GITHUB_TOKEN%20
        publish_dir: ./outputs
        keep_files: true
```

3. Requirements File (src/requirements.txt):

```txt
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
h5py>=3.6.0
jupyter>=1.0.0
papermill>=2.3.0
pandas>=1.3.0
pyyaml>=6.0
gwpy>=2.1.0  # For real LIGO data
astropy>=5.0  # For unit conversions
```

4. Jupyter Notebook Template with Interactive Widgets:

```python
# notebooks/01_simulation_basics.ipynb

import sys
sys.path.append('../src')
from echo_detection_system import EchoDetectionSystem
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown

# Parameters (can be overridden by papermill)
total_mass = 30.0  # Solar masses
detector = 'aLIGO'
n_templates = 100

# Initialize system
system = EchoDetectionSystem(output_dir="../outputs")

# Run simulation
results = system.run_simulation(
    detector=detector,
    f_low=20.0,
    f_high=500.0,
    snr_threshold=8.0,
    n_templates=n_templates,
    inject_signal=True,
    injection_params={
        'Mtot': total_mass,
        'epsilon': 1e-5,
        'R_s_mag': 0.7,
        'R_s_phase': 0.3*np.pi
    }
)

# Display results
display(Markdown(f"## Simulation Results"))
display(Markdown(f"- **Detector**: {results['detector']}"))
display(Markdown(f"- **Templates used**: {len(results['search_results'])}"))
display(Markdown(f"- **Candidates found**: {results['candidates_found']}"))
display(Markdown(f"- **Best candidate SNR**: {results['candidates'][0]['snr'] if results['candidates'] else 'N/A'}"))

# Save to file
import json
with open(f'../outputs/simulation_{total_mass}M_{detector}.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
```

5. GitHub Pages Integration:

Add this to your README.md:

```markdown
# Echo Detection System

[![Run Simulation](https://github.com/yourusername/echo-detection-system/actions/workflows/run-simulation.yml/badge.svg)](https://github.com/yourusername/echo-detection-system/actions/workflows/run-simulation.yml)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yourusername/echo-detection-system/HEAD)

## Live Results
Latest simulation results: [View on GitHub Pages](https://yourusername.github.io/echo-detection-system/)

## Run Your Own Simulation
Click the "Run Simulation" button above or use the manual trigger.
```

6. Automation with GitHub Actions Secrets:

For real LIGO data access, set up secrets in GitHub Settings:

· LIGO_USERNAME
· LIGO_PASSWORD
· LIGO_DATA_DIR
Key features:

· Use EchoDetectionSystem.run_simulation() for quick simulations
· Use EchoDetectionSystem.run_real_data() for real LIGO data (requires GWpy)
· Use EchoDetectionSystem.run_merlin_tests() to run only the Merlin test suite with specified test numbers
· All results automatically saved with timestamps
· Comprehensive error handling

For the Merlin test suite specifically, you can now run only the tests you need:

```python
system.run_merlin_tests(analyzer_results, bridge_data, test_numbers=[1, 2, 3, 4])
```

This will print only the results for tests 1-4, without running all the analysis code