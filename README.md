# ML_with_Graphs-Cyber_Project

**When Zero-Day Becomes Training Data: Learning-Curve Accuracy Stability under Out-of-Distribution Adaptation with GINs on the MalNet-Tiny dataset**
By: Shai Habi, Itamar Link and Tomer Gal Netser

# Introduction
This repository contains both the code implementations and the results for our project research.
In this research, we analyze when and how reliably a GNNs learns to classify Android Function Call Graphs (FCGs) as benign or malwares on MalNet-Tiny.
Using an out-of-distribution approach, we simulate an unseen zero-day malware type and ask how many labeled samples from it must be added to the training set before the model reliably classifies its remaining samples as malicious.

### 1. Research Content:
Our research question is when GINs start converging, namely when their learning-curve accuracy and F1 rates become stable (convergence) on MalNet-Tiny datasets [Freitas et al. 2021], [Tran et al. 2026]. Specifically, we conduct:
1. Incremental OOD Analysis
2. Size-Generalization Analysis

### 2. Datasets and References:
1.Scott Freitas, Yuxiao Dong, Joshua Neil, and Duen Horng Chau. 2021. A Large-Scale Database for Graph Representation Learning. https://arxiv.org/abs/2011.07682
2. Ngoc N. Tran, Anwar Said, Waseem Abbas, Tyler Derr, and Xenofon D. Koutsoukos. 2026. Quantifying the Generalization Gap: A New Benchmark for Out-of-Distribution Graph-Based Android Malware Classification. https://arxiv.org/abs/2508.06734


# Setup
