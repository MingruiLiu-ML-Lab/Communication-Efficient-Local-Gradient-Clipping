# A Communication-Efficient Distributed Gradient Clipping Algorithm for Training Deep Neural Networks

This repository contains PyTorch codes for the experiments on deep learning in the paper:

**[A Communication-Efficient Distributed Gradient Clipping Algorithm for Training Deep Neural Networks](https://openreview.net/forum?id=uLhKRH-ovde)**  
Mingrui Liu, Zhenxun Zhuang, Yunwen Lei, Chunyang Liao.
36th Conference on Neural Information Processing Systems, 2022.

### Description
In distributed training of deep neural networks, people usually run Stochastic Gradient Descent (SGD) or its variants on each machine and communicate with other machines periodically. However, SGD might converge slowly in training some deep neural networks (e.g., RNN, LSTM) because of the exploding gradient issue. Gradient clipping is usually employed to address this issue in the single machine setting, but exploring this technique in the distributed setting is still in its infancy: it remains mysterious whether the gradient clipping scheme can take advantage of multiple machines to enjoy parallel speedup. The main technical difficulty lies in dealing with nonconvex loss function, non-Lipschitz continuous gradient, and skipping communication rounds simultaneously. In this paper, we explore a relaxed-smoothness assumption of the loss landscape which LSTM was shown to satisfy in previous works, and design a communication-efficient gradient clipping algorithm. This algorithm can be run on multiple machines, where each machine employs a gradient clipping scheme and communicate with other machines after multiple steps of gradient-based updates. Our algorithm is proved to have $O\left(\frac{1}{N\epsilon^4}\right)$ iteration complexity and $O(\frac{1}{\epsilon^3})$ communication complexity for finding an $\epsilon$-stationary point in the homogeneous data setting, where $N$ is the number of machines. This indicates that our algorithm enjoys linear speedup and reduced communication rounds. Our proof relies on novel analysis techniques of estimating truncated random variables, which we believe are of independent interest. Our experiments on several benchmark datasets and various scenarios demonstrate that our algorithm indeed exhibits fast convergence speed in practice and thus validates our theory.

### Usage
Please enter each folder to see detailed instructions on how to run each experiment.


### Citation
If you find this repo helpful, please cite the following paper:

```
@inproceedings{liu2022communication,
  title={A Communication-Efficient Distributed Gradient Clipping Algorithm for Training Deep Neural Networks},
  author={Liu, Mingrui and Zhuang, Zhenxun and Lei, Yunwei and Liao, Chunyang},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS)},
  year={2022}
}

```
