# Simulation to Edge AI: An end to end PyTorch deployment pipeline for Arduino Ventuno Q

This project is my research project that will be capstone of my internship at Qualcomm. We want to show how simulation can be used to train and deploy robotics models on the edge. 

## Hardware

The robot that is used is the Turtlebot 4 lite. It is based on the Create3 robot from iRobot, and has a 2D lidar and a OAK-D lite camera.

The robot comes with a Raspberry Pi stock, but we have replaced it with a Arduino Ventuno Q, an unreleased SBC that runs Ubuntu 24.04 LTS. It has a NPU that can be used for AI inference, alongside with an ARM CPU. We have been able to successfully run AI models on the NPU using Executorch. Ask for references to past projects to understand how this was done. 

I will access to two main computer when working on this: A lab desktop with an Nvidia 4090, and a personal desktop with an AMD 6900 XT. Since I should have equal access to both, this shouldn't matter too much, but keep it in mind. 

## Task

The main task is for the robot navigate its 2D environment to some coordinates relative to its starting position, while avoiding obstacles in its environment.

## Simulation

The primary simulation platform is going to be Isaac Sim, along with Isaac Lab. We will also be using Maniskill as well since it is compatible with non-Nvidia GPUs. The framework that we build should be interoperable with these platforms. 

## Training

This is something that you have some freedom to decide over. I initally though of a hybrid training approach with both IL and RL. Here is a plan for that:

1. **Step 1: Bootstrapping via Imitation Learning (IL):**
    
    - Write an oracle path planner (e.g., A* or Fast Marching Method) with access to the simulator's ground-truth map.
        
    - Record $10,000+$ successful trajectories consisting of tuples of $(s_t, a^{\text{expert}}_t)$, where $s_t$ contains the sensor views and the goal vector.
        
    - Train your neural network's perception encoders and policy head using Supervised Behavior Cloning to minimize the Mean Squared Error (MSE) between the predicted velocity command and the expert action:
        
        $$\mathcal{L}_{\text{BC}} = \frac{1}{N}\sum_{i=1}^{N} \vert{}\vert{}\pi_\theta(s_t) - a^{\text{expert}}_t\vert{}\vert{}^2$$
        
2. **Step 2: Reinforcement Learning Fine-Tuning (PPO):**
    
    - Initialize the network weights with the trained IL model.
        
    - Run Proximal Policy Optimization (PPO). Use **Decentralized Distributed PPO (DD-PPO)** if training in Habitat, which scales near-linearly across multiple GPUs.
        
    - **Reward Function:**
        
        $$R_t = \lambda_{\text{goal}} \cdot \mathbb{I}(\text{reached}) + \lambda_{\text{progress}} (d_{t-1} - d_t) - \lambda_{\text{collision}} \cdot \mathbb{I}(\text{collision}) - \lambda_{\text{smooth}} \vert{}\vert{}a_t - a_{t-1}\vert{}\vert{}^2$$
        
        _Where $d_t$ is the geodesic distance to the goal, and the smoothness penalty prevents the physical robot from violently jerking._

Before following this plan, first validate if it is the best way to achieve our goal of training an end-to-end robot navigation model.

## Deployment

Here is the high-level deployment plan that I had AI write up:

Taking your trained policy from a PyTorch-heavy desktop workstation to an embedded robot chassis requires a dedicated deployment pipeline.

1. **Model Export & Optimization:**
    
    - **ONNX Export:** Export the trained PyTorch actor network to ONNX format.
        
    - **Quantization:** Apply Post-Training Quantization (PTQ) to convert the network weights from FP32 to INT8. This is crucial for running inference on low-power NPUs or DSPs.
        
    - **Runtime Compilation:** Use **ExecuTorch** or **TensorRT** (depending on whether your edge platform is CPU/NPU or NVIDIA Jetson-based) to compile the model into an optimized binary file (`.pte` or `.engine`).
        
2. **Inference Integration (ROS 2 C++ Node):**
    
    - Write a high-performance ROS 2 node in C++ to manage high-speed sensor streams.
        
    - **Synchronizer:** Use a message filter to approximate-time synchronize the incoming `/camera/image_raw`, `/scan` (LiDAR), and `/tf` (or goal coordinate topic).
        
    - **Inference Loop:** Feed the synchronized, pre-processed inputs (normalized images and array slices) into the compiled network at a steady 20 Hz.
        
    - **Safety Layer (The Classical Watchdog):** Never map the raw output of a neural network directly to physical motor controllers without a safety wrapper. Implement a simple classical watchdog (like a local obstacle costmap or emergency stop) that overrides the network's velocity commands if a collision is imminent.

Feel free to modify. 