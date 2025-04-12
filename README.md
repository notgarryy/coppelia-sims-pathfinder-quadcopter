# Pathfinder Quadcopter in CoppeliaSim

## Overview

This project aims to develop and simulate a **pathfinding quadcopter** using the **CoppeliaSim** environment. The goal is to enable the quadcopter to navigate through various obstacles in a simulated environment based on a pre-trained model. The model utilizes **convolutional neural networks (CNN)** for image-based decision-making, which allows the quadcopter to autonomously plan a path and avoid obstacles.

This project leverages the **CoppeliaSim Remote API**, and much of the foundational Python integration with CoppeliaSim is based on the work by [JosephLahiru](https://github.com/JosephLahiru/coppeliasim-python). 

## Key Features:
- **Quadcopter Simulation**: Simulated in CoppeliaSim using custom models and paths.
- **Pathfinding Algorithm**: A quadcopter pathfinding system for obstacle avoidance.
- **Deep Learning Model**: CNN-based model to guide the quadcopter's decision-making.
- **Integration with CoppeliaSim**: The quadcopter is controlled through CoppeliaSim's API, providing realistic simulation results.
  
## Prerequisites

- **CoppeliaSim**
- **Python** (with necessary libraries):
  - TensorFlow
  - Keras
  - OpenCV (for image processing)
  - NumPy
- **CoppeliaSim Remote API**

## Note

This project is part of the **mid-term project** for a **robotics course** at university. It demonstrates the integration of deep learning for autonomous navigation in a simulated environment. 

[notgarryy]
