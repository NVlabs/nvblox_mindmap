Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Task/Domain:                                                                   |  Robotic Manipulation
Model Type:                                                                                            |  Denoising Diffusion Probabilistic Model
Intended Users:                                                                                        |  Roboticists and researchers in academia and industry who are interested in robot manipulation research
Output:                                                                                                |  Actions consisting of end-effector poses, gripper states and head orientation.
(For GPAI Models): Tools used to evaluate datasets to identify synthetic data and ensure data authenticity. | Not Applicable
Describe how the model works:                                                                          |  ``mindmap`` is a Denoising Diffusion Probabilistic Model that samples robot trajectories conditioned on sensor observations and a 3D reconstruction of the environment.
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  |  Not Applicable
Technical Limitations & Mitigation:                                                                    |  - The policy is only effective on the exact simulation environment it was trained on. - The policy was never tested on a physical robot and likely only works in simulation.
Verified to have met prescribed NVIDIA quality standards:  |  Yes
Performance Metrics:                                                                                   |  Closed loop success rate on simulated robotic manipulation tasks.
Potential Known Risks:                                                                                 |  The model might be susceptible to rendering changes on the simulation tasks it was trained on.
Licensing:                                                                                             |  [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
