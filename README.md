# AI-VarDA: A Unified Variational Data Assimilation Framework for AI and Traditional Methods

This framework builds on our recent works, [FengWu-4DVar](https://openreview.net/forum?id=Y2WorV5ag6) and [VAE-Var](https://openreview.net/forum?id=utz99dx2RN), aiming to bridge traditional data assimilation techniques with modern deep generative modeling. **VAR-AI** is a modular and extensible Python-based framework that supports both classic variational methods (e.g., 3DVar, 4DVar, GEN_BE) and AI-driven approaches (e.g., VAE-Var, LoRA-EnVar). It offers a unified interface for integrating forecast models, background error representations, and observation operators — enabling flexible configuration, comparative experimentation, and reproducible research.



## 🌟 Key Features

- ✅ **Hybrid support** for AI-based and traditional DA algorithms
- 🔌 **Pluggable architecture**: swap decoders, models, and observation operators
- 📈 **Integrated support** for ensemble-based flow error propagation
- 🧠 **Learnable background error models** via VAE or GEN_BE-style approaches (See )
- 💡 Designed for **weather forecasting**, but extensible to other geophysical applications



## 🗂️ Project Structure Overview

```
assimilation_framework/
├── assimilation_core/           # Core data assimilation logic
│   ├── assimilation_models/     # All DA algorithm implementations (3DVar, 4DVar, VAE-Var, etc.)
│   ├── bg_decoder_models/       # Control-to-physical variable decoders (linear, VAE, GEN_BE)
│   ├── data_reader/             # Physical field and observation data loaders
│   ├── flow_models/             # Forecast model for 4DVar flow dependendies
│   ├── observation_operator/    # Observation operators (identity, interpolation, etc.)
│   ├── ensemble_generator.py    # Ensemble background propagation
│   ├── forecast_model.py        # Forecast model runner interface
│   ├── init_states_constructor.py  # Initial ensemble state generator
│   ├── evaluator.py             # Evaluation: error computation, diagnostics
│   └── runner.py                # Main assimilation process controller

├── bgerr_learning/              # Static background error training modules
│   ├── dataset/                 # ERA5-based datasets and normalization
│   ├── learning_algorithms/     # GEN_BE and VAE-BE learners
│   ├── utils/                   # Dataset builder utilities
│   └── runner.py                # Training entry point

├── networks/                    # Deep learning architectures
│   ├── bg_vae/                  # Background VAE model
│   ├── fengwu_hr/               # FengWu high-resolution UNet + attention variants
│   ├── fengwu_lr/               # Transformer-based FengWu low-resolution model

├── utils/                       # Logging, metrics, helper functions
│   ├── logger.py
│   ├── metrics.py
│   └── misc.py

scripts/                         # Python entry scripts
├── run_assimilation_loop.py     # Online assimilation cycle
└── learn_background_error.py    # Offline background error model training

bin/                             # Shell launch scripts (e.g., SLURM)
├── run_assimilation_loop.sh
├── run_genbe_learn_error.sh
└── run_vaebe_learn_error.sh

config/                          # YAML-based experiment and model configs
├── assimilation_loop/           # Configs for DA cycle
└── bgerr_learning/              # Configs for GEN_BE and VAE training

checkpoints/                     # Pretrained models and intermediate outputs
├── bgerr_models/                # Trained background error models (GEN_BE, VAE)
├── forecast_models/             # FengWu model checkpoints
├── flow_error/                  # Learned ensemble perturbation statistics
└── observation_masks/           # Observation availability masks

experiments/                     # Output directory for experiment logs and results
├── assimilation/
└── learning_background/

README.md
```



## 🚀 Quick Start

This framework supports both **static background error learning** and **online data assimilation** via configurable YAML files.
 We provide three runnable shell scripts in `bin/` for common workflows:

### 1. Online Data Assimilation

Run the end-to-end data assimilation loop (e.g., 3DVar, VAE-4DVar, LoRA-EnVar):

```
bash bin/run_assimilation_loop.sh
```

This script launches a sequence of predefined experiments defined in:

- `config/assimilation_loop/exp_*.yaml`

The results and logs will be saved under:

```
experiments/assimilation/<experiment_prefix>/
```

------

### 2. Static Background Error Learning

#### GEN_BE-style learning:

```
bash bin/run_genbe_learn_error.sh
```

#### VAE-based learning:

```
bash bin/run_vaebe_learn_error.sh
```

Both use configurations in:

- `config/bgerr_learning/`

And save output to:

```
experiments/learning_background/<prefix>/
```

### 3. Customization

You can create or modify your own YAML config files to define:

- Assimilation algorithm (`gaussian_var`, `vae_var`, `lora_envar`, etc.)
- Forecast model and resolution
- Observation scenarios (random, gridded, interpolated)
- Background error representation (GEN_BE, VAE)
- Ensemble settings, time window size, training parameters, etc.

> 📁 See `config/assimilation_loop/` and `config/bgerr_learning/` for examples.

Then, update the corresponding shell script or launch manually:

```
python scripts/run_assimilation_loop.py --config your_config.yaml --prefix your_run_name
```



## 📄 Related Publications

This framework implements and extends the following research works:

- **FengWu-4DVar**: Xiao et al., *FengWu-4DVar: Coupling the Data-Driven Weather Forecasting Model with 4D Variational Assimilation*. [[conference link](https://openreview.net/forum?id=Y2WorV5ag6)] [[arxiv link](https://arxiv.org/abs/2312.12455)]

  > Introduces a differentiable AI-forecast-driven 4DVar system capable of stable 1-year cycling 

- **VAE-Var**: Xiao et al., *VAE-Var: Variational-Autoencoder-Enhanced Variational Assimilation in Meteorology*. [[Conference link](https://openreview.net/forum?id=utz99dx2RN)]

  > Proposes a novel method to model background error distributions using deep generative models, enabling assimilation of off-grid observations.



## 📌 Citation

If you use this framework or build upon FengWu-4DVar or VAE-Var, please cite:

```
@article{xiao2023fengwu,
  title={Fengwu-4dvar: Coupling the data-driven weather forecasting model with 4d variational assimilation},
  author={Xiao, Yi and Bai, Lei and Xue, Wei and Chen, Kang and Han, Tao and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2312.12455},
  year={2023}
}

@inproceedings{xiao2024towards,
  title={Towards a self-contained data-driven global weather forecasting framework},
  author={Xiao, Yi and Bai, Lei and Xue, Wei and Chen, Hao and Chen, Kun and Han, Tao and Ouyang, Wanli and others},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}

@inproceedings{xiao2025vae,
  title={VAE-Var: Variational autoencoder-enhanced variational methods for data assimilation in meteorology},
  author={Xiao, Yi and Jia, Qilong and Chen, Kun and Bai, Lei and Xue, Wei},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```



## ✅ TODO

We are actively developing and expanding the framework. Below is a roadmap of planned features:

### 🔜 Short-Term Goals

-  **Support for real-world station observations**
   Integration of non-gridded, irregularly located surface station data into the assimilation pipeline (via configurable observation operators).

### 🛠️ Long-Term Plans

-  **Implement additional traditional DA algorithms**
   Including hybrid EnKF-Var, weak-constraint 4DVar, and flow-dependent covariance models.
-  **Integrate with recent AI-based assimilation methods**
   Coupling with models like [DiffDA](https://arxiv.org/abs/2401.05932), [APPA](https://arxiv.org/abs/2504.18720), and other diffusion-based or neural inverse solvers.
-  **Assimilation of satellite and radar observations**
   Building generalized observation operators and preprocessing pipelines for satellite radiances and radar reflectivity.

*Feel free to open an issue or pull request if you'd like to contribute to any of these features!*

Maintainer: [Yi Xiao](mailto:xiaoyi200018@gmail.com)



