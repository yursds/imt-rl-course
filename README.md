# 🎮 IMT Reinforcement Learning Course Repository

![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)

This repository contains all assignments and resources for the **Reinforcement Learning (RL) course** at IMT Lucca, helded by Mario Zanon. It includes implementations of core RL algorithms, training videos, pre-trained models, and Jupyter notebooks with explanations.

---

## 📂 Repository Structure

```bash
.
├── assign_*-*.[ipynb|html]       # Assignment notebooks
├── assign-pdf                    # Assignment texts
│   └── ...   
├── img/                          # Figures 
│   ├── assign-1/                 
│   ├── assign-2/                 
│   └── ...                      
├── saved_models/                 # Pre-trained model parameters
├── video/                        # Training and test videos
│   └── ...                      
├── jax_utils_stuff/              # Environment simulators and JAX utilities   
│   └── ...
├── pyproject.toml                # Python project metadata
└── README.md                     
```

---

## 🚀 Assignments

| Assignment | Topics |
|------------|--------|
| **1**      | Policy Evaluation, Value Iteration |
| **2**      | Monte Carlo, SARSA, Q-Learning |
| **3**      | Deep Q-Networks (DQN), Function Approximation |
| **4**      | Policy Gradients (REINFORCE, SAC) |

**Notebook Features**:
- Interactive code cells
- Algorithm pseudocode (e.g., SARSA, DQN)
- Training progress visualizations

---

## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/assignments-imt-rl.git
cd assignments-imt-rl
```

### 2. Install Dependencies
```bash
# Using pip
pip install -r requirements.txt

# Or with uv (faster)
uv sync
```

### 3. Run Jupyter Notebooks
```bash
jupyter notebook
# Open assign_X-Y.ipynb and follow the instructions
```

---

## 🖼️ Key Directories

### `img/`
- **Purpose**: Stores plots and diagrams for reports.
- **Example**: `assign-1/1-a.png` shows the policy evaluation results for Assignment 1.

### `saved_models/`
- **Usage**: Load pre-trained models to skip training:
  ```python
  import numpy as np
  params = np.load('saved_models/3/model_params_DQN.npz')
  ```

### `video/`
- **Content**: Recordings of agent behavior during training/testing.
- **Example**: `video/4-3a/tabular/wo_baseline/` compares REINFORCE with/without baselines.

### `jax_utils_stuff/`  
- **[Tools](https://mariozanon.wordpress.com/wp-content/uploads/2025/02/useful_files-1.pptx)**: 
  - `frozen_lake_sim.py`: Custom Frozen Lake environment.
  - `jax_haiku.py`: Neural network builders using Haiku.


---

## 🤝 Contribution

Contributions are welcome! To add/fix content:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

---


## 📧 Contact

For issues:
- **GitHub Issues**: [Open an Issue](https://github.com/your-username/assignments-imt-rl/issues)