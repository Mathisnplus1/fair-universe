# An insight into the Fair Universe project

## Background and motivation

---

The Fair Universe project at LBNL is dedicated to creating an AI competition geared towards mitigating the impacts of systematic uncertainty in High Energy Physics. High-energy physicists stationed at CERN heavily rely on simulations to replicate the collisions that are observed within the Large Hadron Collider (LHC). These particle collisions produce numerous smaller particles. When they propose a theory predicting the existence of a new particle, physicists use these simulations to search for evidence supporting its presence. For this purpose, they categorize the particles resulting from collisions into background particles (already known and uninteresting) and signal particles (the ones of interest). To perform this classification task, high-energy physicists are increasingly collaborating with machine learning scientists. A wealth of features are known about each particle, such as speed, energy, and angle measurements. Nevertheless, due to imperfection in the simulators, the simulated data are sensitive to systematic biases, making the classification task more challenging. Consequently, a major obstacle is to eliminate these biases from the data to enhance classification. To tackle this issue, the Fair Universe project aims at building an online challenge.

The Fair Universe's toy-challenge serves as a simplified framework of this problem. We have designed a straightforward simulator that aims to resemble a physics scenario and employs Gaussian and Gamma distributions. In order to avoid working within a high-dimensional feature space, particles are simulated as 2D points, classified into either the signal or background class. The goal is to develop models capable of accurately classifying these points. The biases affecting the particles are represented as combinations of translation, rotation and scaling that affect all the points, regardless of their class. In the following sections, we explore domain adaptation techniques to establish a performance benchmark for evaluating challenge submissions. This is needed to identify relevant solutions proposed by contestants that outperform our own.

***

The official GitHub repository for the Fair Universe Toy Challenge is https://github.com/ihsaan-ullah/fair-universe

___

### In this repository