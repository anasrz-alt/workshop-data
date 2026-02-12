To accommodate the Kaggle dataset requirement in **Notebook 2**, I have added a "Setup & Authentication" section to the README. This ensures participants have their API credentials ready before starting the workshop.

---

# Workshop 5.3

## Overview

This workshop explores the intersection of classical engineering and deep learning to solve real-world constraints in robotics and distributed computing. From compensating for hardware latency in quadrotors to managing massive GenAI workloads across clusters, these notebooks demonstrate how to deploy "intelligent" logic that is both fast enough for the edge and robust enough for production.

---

## 📚 Notebooks

### 1. 01_Latency_Aware_Control.ipynb - PI-LSTM Compensation

**Focus:** Mitigating communication delays in high-speed robotics using Physics-Informed LSTMs.
**Key Features:**

* **Hybrid Modeling:** Combines physics-based projection (velocity) with an LSTM residual error predictor.
* **Control Pipeline:** Integration of Model Predictive Control (MPC) with neural corrections.
* **Metric Tracking:** Quantitative evaluation using Euclidean error and 3D trajectory visualization.

**Learning Outcomes:**

* Understand how predictive modeling compensates for hardware "overshoot."
* Implement a hybrid physics + neural network control loop.

### 2. 02_Imitation_Learning.ipynb - End-to-End Robotics

**Focus:** Mapping raw sensor data directly to motor commands for autonomous navigation.
**Key Features:**

* **Sensor Abstraction:** Binning high-dimensional LiDAR point clouds into simplified spatial cones (Front/Left/Right).
* **Behavioral Cloning:** Using Supervised Learning (MSE loss) to mimic expert agent maneuvers.
* **Real-World Data:** Training on the Kaggle IRND dataset to handle sensor jitter and varying terrains.

**Learning Outcomes:**

* Apply "Perception-to-Action" mapping without manual if-then rules.
* Implement robust data pipelines for physical sensor telemetry.

### 3. 03_Intelligent_Offloading.ipynb - GenAI Cluster Scheduling

**Focus:** Building a "Policy Network" to manage distributed Generative AI inference tasks.
**Key Features:**

* **Production Data:** Utilizes real-world telemetry from the Alibaba GenAI Cluster Trace (2026).
* **Cost Scoring:** Implements a median-based labeling strategy to balance Local vs. Offload decisions.
* **SLA Adherence:** Evaluation using confusion matrices to minimize network-induced latency.

**Learning Outcomes:**

* Articulate trade-offs between local GPU utilization and network congestion.
* Build a neural decision engine for bursty, non-linear traffic patterns.

---

## 🏗️ Technical Details

### Hardware & Data Contexts

* **Robotics (NB 1 & 2):** Focused on low-latency, high-frequency control loops where "Sim-to-Real" gaps are critical.
* **Distributed Systems (NB 3):** Focused on resource throughput and minimizing "bottlenecks" in GPU clusters.

---

## 🚀 Getting Started

### 1. Kaggle API Setup (Required for Notebook 2)

To download the **Indoor Robot Navigation Dataset (IRND)**, you must provide your Kaggle credentials:

1. Log in to [Kaggle](https://www.kaggle.com/).
2. Go to **Settings** and click **Create New Token** to download `kaggle.json`.
3. Open `02_Imitation_Learning.ipynb` and enter your `KAGGLE_USERNAME` and `KAGGLE_KEY` in the first cell:

```python
import os
os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY'] = "your_api_key"

```

### 2. Prerequisites

On Google Colab, these notebooks work without needing any setup.

```bash
pip install torch torchvision matplotlib numpy pandas scikit-learn tqdm kaggle

```

---

## 📊 Key Insights

* **The Hybrid Edge:** Pure deep learning is often too slow for the edge; combining it with physics (NB 1) or simplified abstractions (NB 2) is the key to real-time performance.
* **Data-Driven Scheduling:** In the era of GenAI, static rules are too brittle. Intelligent offloading (NB 3) allows infrastructure to adapt to traffic spikes dynamically.
