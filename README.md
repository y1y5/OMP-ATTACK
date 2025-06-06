<div align="center">
  
# Enduring, Efficient and Robust Trajectory Prediction Attack in Autonomous Driving via Optimization-Driven Multi-Frame Perturbation Framework

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![Conference](https://img.shields.io/badge/CVPR-2025-blue)](https://cvpr.thecvf.com/Conferences/2025)
</div>

## Updates
<!-- :satisfied: (03/11/2024) Code Released! -->
- (02/27/2025) Paper Accepted!

## Abstract

Trajectory prediction plays a crucial role in autonomous driving systems, and exploring its vulnerability has garnered widespread attention. However, existing trajectory prediction attack methods often rely on single-point attacks to make efficient perturbations. This limits their applications in real-world scenarios due to the transient nature of single-point attacks, their susceptibility to filtration, and the uncertainty regarding the deployment environment. To address these challenges, this paper proposes a novel LiDAR-induced attack framework to impose multi-frame attacks by optimization-driven adversarial location search, achieving endurance, efficiency, and robustness. This framework strategically places objects near the adversarial vehicle to implement an attack and introduces three key innovations. First, successive state perturbations are generated using a multi-frame single-point attack strategy, effectively misleading trajectory predictions over extended time horizons. Second, we efficiently optimize adversarial objects' locations through three specialized loss functions to achieve desired perturbations. Lastly, we improve robustness by treating the adversarial object as a point without size constraints during the location search phase and reduce dependence on both the specific attack point and the adversarial object's properties. Extensive experiments confirm the superior performance and robustness of our framework.

## Attack Steps

### Setup
1. Clone this repository.
2. CD to the `PIXOR_nuscs` directory and create a conda environment for the PIXOR detector:
   ```bash
   # create env
   cd PIXOR_nuscs
   conda create -n pixor_nuscs python=3.7

   # install dependecies
   Follow [PIXOR](https://github.com/philip-huang/PIXOR) to install the environment
   ```
3. CD to the `Trajectron-plus-plus` directory and create a conda environment for the Trajectron++ predictor:
   ```bash
   # create env
   cd Trajectron-plus-plus
   conda create -n trajectron++ python=3.7

   # install dependecies
   Follow [Trajectron-plus-plus](https://github.com/StanfordASL/Trajectron-plus-plus) to install the environment
   ```

### Data preparation
  Download the Full dataset (V1.0) of the nuScenes dataset from [nuScenes](https://www.nuscenes.org/nuscenes#download).

### Training of PIXOR detector and Trajectron++ predictor
> You can follow the official guide to train your own models. Then replace the models in `OMP-ATTACK/PIXOR_nuscs/srcs/experiments/nusce_kitti` and `OMP-ATTACK/Trajectron-plus-plus/experiments/nuScenes/models/int_ee` with your own models.

### Attack
- Initialize paths and files in `./omp-attack.sh`, including 'root_dir', 'target_scene_instance_file', 'code_dir', and 'nusc_datadir'.
- Follow the steps in `./omp-attack.sh` for attack and evaluation.

## Citation
If you find this paper or the code useful for your research, please consider citing:
```bibtex
@inproceedings{yu2025enduring,
  title={Enduring, Efficient and Robust Trajectory Prediction Attack in Autonomous Driving via Optimization-Driven Multi-Frame Perturbation Framework},
  author={Yu, Yi and Han, Weizhen and Wu, Libing and Liu, Bingyi and Wang, Enshu and Zhang, Zhuangzhuang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17229--17238},
  year={2025}
}
```

## Acknowledgments

Our code is developed based on [Lou, et al. "A First {Physical-World} Trajectory Prediction Attack via {LiDAR-induced} Deceptions in Autonomous Driving." 33rd USENIX Security Symposium (USENIX Security 24).](https://github.com/kyrie-louy/physical-traj-pred-attack) Thanks for their contributions and useful help.
