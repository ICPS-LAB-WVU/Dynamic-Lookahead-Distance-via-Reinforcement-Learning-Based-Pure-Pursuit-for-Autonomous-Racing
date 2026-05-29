# Dynamic Lookahead Distance via Reinforcement Learning-Based Pure Pursuit for Autonomous Racing

**Author:** Mohamed Elgouhary
**Affiliation:** iCPS Lab, West Virginia University
**Project Type:** ROS 2 / F1TENTH Autonomous Racing Controller

This repository provides code, trained models, and example files related to the paper:

> **Dynamic Lookahead Distance via Reinforcement Learning-Based Pure Pursuit for Autonomous Racing**
> Mohamed Elgouhary and Amr S. El-Wakeel
> arXiv:2603.28625, 2026
> DOI: https://doi.org/10.48550/arXiv.2603.28625

## Overview

Pure Pursuit is a widely used path-tracking controller for autonomous vehicles and F1TENTH-style autonomous racing because it is simple, interpretable, and computationally efficient. However, its performance depends strongly on the selected lookahead distance.

A short lookahead distance can improve cornering accuracy but may cause aggressive or unstable behavior on straights. A long lookahead distance can improve smoothness but may reduce tracking accuracy in curves.

This project addresses this limitation by using **reinforcement learning** to dynamically tune the Pure Pursuit lookahead distance during racing.

The proposed framework keeps the classical Pure Pursuit controller structure and uses a learned **Proximal Policy Optimization (PPO)** policy to adjust only one interpretable parameter:

```text
Dynamic lookahead distance
```

This makes the controller lightweight, interpretable, and easier to integrate into modular ROS 2 autonomous racing stacks.

## Key Idea

Instead of replacing Pure Pursuit with an end-to-end neural-network controller, this method augments Pure Pursuit with a learned lookahead policy.

At each control step:

1. The vehicle follows a reference raceline.
2. The system extracts compact driving features such as vehicle speed and upcoming path curvature.
3. A PPO policy predicts a suitable lookahead distance.
4. The predicted lookahead distance is passed to the Pure Pursuit controller.
5. Pure Pursuit computes the steering command.
6. The vehicle tracks the raceline using the dynamically adjusted lookahead.

This preserves the structure and interpretability of Pure Pursuit while improving adaptability across straights, curves, and unseen tracks.

## Repository Contents

```text
Dynamic-Lookahead-Distance-via-Reinforcement-Learning-Based-Pure-Pursuit-for-Autonomous-Racing/
├── README.md
└── src/
    ├── best_model/
    ├── csv_data/
    ├── new/
    ├── old/
    ├── paper2_pkg/
    ├── paper_pkg/
    ├── test/
    ├── test2/
    ├── Montreal.png
    ├── Montrel.csv
    ├── ppo_lookahead_model.zip
    └── vecnorm.pkl
```

## Main Components

| Path                          | Description                                                                             |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| `src/paper_pkg/`              | ROS 2 package containing the Pure Pursuit / dynamic-lookahead controller implementation |
| `src/paper2_pkg/`             | Additional ROS 2 package or experimental version used during development                |
| `src/csv_data/`               | Raceline and track CSV files                                                            |
| `src/best_model/`             | Saved best PPO model or training checkpoint files                                       |
| `src/ppo_lookahead_model.zip` | Trained PPO policy for dynamic lookahead prediction                                     |
| `src/vecnorm.pkl`             | Normalization statistics used with the trained PPO policy                               |
| `src/Montrel.csv`             | Example Montreal raceline/track file                                                    |
| `src/Montreal.png`            | Example Montreal track image                                                            |
| `src/new/`, `src/old/`        | Development versions or archived experimental files                                     |
| `src/test/`, `src/test2/`     | Testing or experimental scripts                                                         |

## Method Summary

The proposed controller uses a learning-augmented Pure Pursuit structure:

```text
Vehicle state + raceline features
        ↓
PPO lookahead policy
        ↓
Dynamic lookahead distance
        ↓
Pure Pursuit controller
        ↓
Ackermann steering command
```

The PPO policy is trained to adjust the lookahead distance online using compact features such as:

* Vehicle speed
* Near-horizon raceline curvature
* Mid-horizon raceline curvature
* Far-horizon raceline curvature

The output is a dynamic lookahead command used by the Pure Pursuit controller.

## Why This Repository Is Useful

This repository can help researchers, students, and developers working on:

* Autonomous racing
* F1TENTH control
* ROS 2 vehicle control
* Pure Pursuit path tracking
* Reinforcement learning for controller tuning
* PPO-based parameter adaptation
* Sim-to-real autonomous racing
* Learning-augmented classical control
* Dynamic lookahead selection
* Interpretable reinforcement learning for robotics

## Installation

### 1. Clone the repository

Clone this repository into the `src` folder of your ROS 2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ICPS-LAB-WVU/Dynamic-Lookahead-Distance-via-Reinforcement-Learning-Based-Pure-Pursuit-for-Autonomous-Racing.git
```

### 2. Build the workspace

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 3. Install Python dependencies

The exact environment may depend on your ROS 2 and F1TENTH setup. Common Python dependencies include:

```bash
pip install numpy scipy pandas matplotlib stable-baselines3 gym
```

Depending on your simulator version, you may also need:

```bash
pip install torch
```

For ROS 2, make sure the following packages are available:

* `rclpy`
* `nav_msgs`
* `geometry_msgs`
* `sensor_msgs`
* `ackermann_msgs`
* `visualization_msgs`

## Running the Controller

After building and sourcing your ROS 2 workspace, check the available package executables:

```bash
ros2 pkg list | grep paper
ros2 pkg executables paper_pkg
ros2 pkg executables paper2_pkg
```

Then run the appropriate executable shown by ROS 2.

Example:

```bash
ros2 run paper_pkg <executable_name>
```

Replace `<executable_name>` with the executable listed by:

```bash
ros2 pkg executables paper_pkg
```

If your controller is inside `paper2_pkg`, use:

```bash
ros2 run paper2_pkg <executable_name>
```

## Using the Trained PPO Lookahead Policy

The repository includes a trained PPO model:

```text
src/ppo_lookahead_model.zip
```

and the corresponding normalization file:

```text
src/vecnorm.pkl
```

When using the trained policy, make sure that:

1. The PPO model and normalization file are loaded together.
2. The observation format matches the format used during training.
3. The raceline CSV file has the expected columns.
4. The vehicle parameters match the simulator or real F1TENTH car.
5. The ROS 2 topic names match your setup.

A typical inference workflow is:

1. Load the raceline CSV.
2. Read the current vehicle pose and velocity.
3. Compute speed and curvature-based features.
4. Normalize the observation using `vecnorm.pkl`.
5. Predict the dynamic lookahead using `ppo_lookahead_model.zip`.
6. Pass the predicted lookahead to the Pure Pursuit controller.
7. Publish the steering and speed command.

## Track and Raceline Files

The repository includes an example Montreal track image and CSV file:

```text
src/Montreal.png
src/Montrel.csv
```

Additional raceline files can be placed inside:

```text
src/csv_data/
```

To use a new track, prepare a compatible raceline CSV file and update the file path in the controller code or launch configuration.

## Expected ROS 2 Topics

The exact topics depend on your F1TENTH simulator or real-car setup. A typical setup uses:

| Type         | Topic                   | Description                                           |
| ------------ | ----------------------- | ----------------------------------------------------- |
| Input        | `/ego_racecar/odom`     | Vehicle odometry in simulation                        |
| Input        | `/pf/viz/inferred_pose` | Estimated pose on the real vehicle                    |
| Output       | `/drive`                | Ackermann steering and speed command                  |
| Input/Output | `/scan`                 | LiDAR scan, if used by the surrounding stack          |
| Output       | RViz marker topics      | Waypoints, target point, and trajectory visualization |

Before running, verify your topic names using:

```bash
ros2 topic list
```

## Suggested Workflow for New Users

1. Clone the repository into a ROS 2 workspace.
2. Build the workspace with `colcon build`.
3. Source the workspace.
4. Confirm that the ROS 2 package appears in `ros2 pkg list`.
5. Confirm the available executable using `ros2 pkg executables`.
6. Start the F1TENTH simulator or real-car localization stack.
7. Verify odometry and `/drive` topic compatibility.
8. Load the trained PPO model and normalization file.
9. Run the dynamic-lookahead Pure Pursuit controller.
10. Visualize the vehicle trajectory and raceline in RViz.
11. Compare the dynamic-lookahead controller with fixed-lookahead Pure Pursuit.

## Suggested Baselines for Comparison

Researchers can compare this method against:

* Fixed-lookahead Pure Pursuit
* Speed-based adaptive Pure Pursuit
* Curvature-based adaptive Pure Pursuit
* Stanley controller
* LQR path tracking
* Model Predictive Control
* End-to-end reinforcement learning controllers
* PPO-based lookahead and steering-gain tuning

## Suggested Experiments

This repository can be extended by testing:

1. New unseen racetracks.
2. Different speed profiles.
3. Different PPO observation spaces.
4. Different reward functions.
5. Different lookahead limits.
6. Sim-to-real transfer on additional F1TENTH vehicles.
7. Robustness to localization noise.
8. Robustness to raceline perturbations.
9. Comparison with MPC, LQR, Stanley, and MPPI.
10. Safety filters around learned lookahead outputs.

## Reproducibility Notes

To reproduce results as closely as possible:

* Use the provided trained PPO model.
* Use the matching `vecnorm.pkl` file.
* Use the same raceline format.
* Verify the same vehicle parameters.
* Verify the same controller frequency.
* Confirm the same ROS 2 topic names.
* Test in simulation before real-car deployment.
* Start at low speed before increasing the velocity profile.

## Safety Notes

This repository is intended for research and development. Before using the controller on a physical F1TENTH vehicle:

* Test in simulation first.
* Use an emergency stop.
* Start with conservative speed limits.
* Verify steering direction.
* Verify odometry direction.
* Confirm raceline alignment with the map.
* Confirm that the lookahead output remains within safe limits.
* Avoid testing near people or obstacles during initial runs.

## Citation

If you use this repository, the trained model, or the dynamic-lookahead Pure Pursuit method in your research, please cite the associated paper:

```bibtex
@misc{elgouhary2026dynamic,
  title={Dynamic Lookahead Distance via Reinforcement Learning-Based Pure Pursuit for Autonomous Racing},
  author={Elgouhary, Mohamed and El-Wakeel, Amr S.},
  year={2026},
  eprint={2603.28625},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  doi={10.48550/arXiv.2603.28625}
}
```

## Recommended Citation Text

You may cite this work as:

> M. Elgouhary and A. S. El-Wakeel, “Dynamic Lookahead Distance via Reinforcement Learning-Based Pure Pursuit for Autonomous Racing,” arXiv:2603.28625, 2026.

## Paper Links

* arXiv: https://arxiv.org/abs/2603.28625
* DOI: https://doi.org/10.48550/arXiv.2603.28625

## How to Reuse This Work

You can use this repository to:

* Reproduce dynamic-lookahead Pure Pursuit experiments.
* Compare fixed-lookahead and learned-lookahead Pure Pursuit.
* Study how lookahead distance affects racing performance.
* Test learning-augmented classical control ideas.
* Develop new RL-based controller-parameter tuning methods.
* Extend the observation space with additional perception or vehicle-state features.
* Deploy a trained dynamic-lookahead policy in a ROS 2 F1TENTH stack.

## Suggested GitHub Topics

Add these topics to the GitHub repository:

```text
f1tenth
ros2
autonomous-racing
pure-pursuit
dynamic-lookahead
reinforcement-learning
ppo
stable-baselines3
path-tracking
autonomous-driving
sim-to-real
robotics
ackermann
```

## Suggested GitHub Description

Use this as the repository description:

```text
ROS 2 implementation of reinforcement-learning-based dynamic lookahead tuning for Pure Pursuit in F1TENTH autonomous racing.
```

## Authors

**Mohamed Elgouhary**
Lane Department of Computer Science and Electrical Engineering
West Virginia University

**Amr S. El-Wakeel**
Lane Department of Computer Science and Electrical Engineering
West Virginia University

## Acknowledgment

This repository is associated with autonomous vehicle and F1TENTH research at the iCPS Lab, West Virginia University.

This work was partially supported by DARPA AI-CRAFT under Grant AWD16069.

## License

Please add an appropriate license file before public reuse.

If you want the code to be broadly reusable and citable, consider adding an open-source license such as MIT, BSD-3-Clause, or Apache-2.0 after confirming with all co-authors and the lab.

## Contact

For questions about the paper or repository, please contact:

**Mohamed Elgouhary**
Email: [mae00018@mix.wvu.edu](mailto:mae00018@mix.wvu.edu)
