Identifying Influential Nodes In Complex Networks via Transformer

This paper brings several noteworthy contributions to the research field, which can be summarized as follows:

(1) In this paper, the Transformer model is introduced into the complex network to convert the node influence identification task into a regression task.

(2) By customizing the node sequence, the nodes can fuse the information of first-order neighbors and second-order neighbors. Experiments show that the Transformer can achieve the same aggregation effect as GNN, and the Transformer is more stable in complex networks.

(3) The CNT model is different from the previous GCN and GNN-based models in that the length of the input sequence can be dynamically adjusted according to the size of the complex network. Sequence features of different lengths can be input for various complex networks, which allows the CNT model to obtain enough node information so that it can handle complex networks of different sizes.

(4) We performed several parametric analyses to determine the optimal length of the input sequence as a way to obtain the most efficient model.

(5) The CNT model outperforms seven benchmark methods in identifying node influence across 9 synthetic and 12 real-world networks with various infection probabilities.


This repository includes the dataset introduced by the following paper: Adjnoun, Facebook, Netscience, Jazz, Lesmis, Hamster, Moreno, Plobooks, USAir PowerGrid etc.

Model：
![model](https://github.com/cly1022/socialmedia-CNT/assets/17700771/f99bb531-45e0-47aa-8179-cd04a49ea056)


Windows Server 2022 Standard

12th generation Intel(R) Core(TM) i9-12900K   3.2GHz

128 GB

NVIDIA RTX A6000

Python 3.7.6


If you have any questions about the paper and repository, feel free to contact Leiyang Chen (cly_edu@whu.edu.cn) or open an issue!
