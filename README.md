# DirectionNet (TensorFlow 2.x Implementation)

This repository provides the **TensorFlow 2.x** reimplementation of the model introduced in the CVPR 2020 paper:

**Wide-Baseline Relative Camera Pose Estimation with Directional Learning**  
*Kefan Chen, Noah Snavely, Ameesh Makadia*  
Presented at **IEEE CVPR 2020**

- [arXiv Paper](https://arxiv.org/abs/2106.03336)  
- [Original TensorFlow 1.x Repository](https://github.com/arthurchen0518/DirectionNet.git)  

---

## Environment Setup

To reproduce results and run the code, follow these steps:

1. **Create a new conda environment (Python 3.10.18 recommended):**
   ```bash
   conda create -n <your_env_name> python=3.10.18
2.	**Activate the environment:**
   ```bash
   conda activate <your_env_name>

3. **Install dependencies:**
   ```bash
   pip install tensorflow==2.15.1
   pip install keras==2.15.0
   pip install tensorflow-addons==0.23.0
   pip install tensorflow-probability==0.23.0
   pip install tensorflow-graphics==2021.12.3

# **Install additional packages**
   ```bash
   pip install -r requirements.txt


