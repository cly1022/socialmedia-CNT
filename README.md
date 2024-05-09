# Title

Identifying Influential Nodes In Complex Networks via Transformer

##Abstract
In the domain of complex networks, the identification of influential nodes plays a crucial role in ensuring network stability and facilitating efficient information dissemination.
Although the study of influential nodes has been applied in many fields such as suppression of rumor spreading, regulation of group behavior, and prediction of mass events evolution, current deep learning-based algorithms have limited input features and are incapable of aggregating neighbor information of nodes, thus failing to adapt to complex networks.We propose an influential node identification method in complex networks based on the Transformer.
In this method, the input sequence of a node includes information about the node itself and its neighbors, enabling the model to effectively aggregate node information to identify its influence. Experiments were conducted on 9 synthetic networks and 12 real networks.
Using the SIR model and a benchmark method to verify the effectiveness of our approach.The experimental results show that this method can more effectively identify influential nodes in complex networks.
In particular, the method improves 27 percent compared to the second place method in network Netscience and 21 percent in network Faa.

##contributions
This paper brings several noteworthy contributions to the research field, which can be summarized as follows:

(1) In this paper, the Transformer model is introduced into the complex network to convert the node influence identification task into a regression task.

(2) By customizing the node sequence, the nodes can fuse the information of first-order neighbors and second-order neighbors. Experiments show that the Transformer can achieve the same aggregation effect as GNN, and the Transformer is more stable in complex networks.

(3) The CNT model is different from the previous GCN and GNN-based models in that the length of the input sequence can be dynamically adjusted according to the size of the complex network. Sequence features of different lengths can be input for various complex networks, which allows the CNT model to obtain enough node information so that it can handle complex networks of different sizes.

(4) We performed several parametric analyses to determine the optimal length of the input sequence as a way to obtain the most efficient model.

(5) The CNT model outperforms seven benchmark methods in identifying node influence across 9 synthetic and 12 real-world networks with various infection probabilities.

##datesets
This repository includes the dataset introduced by the following paper: Adjnoun, Facebook, Netscience, Jazz, Lesmis, Hamster, Moreno, Plobooks, USAir PowerGrid etc.
URL:[https://networkrepository.com/index.php]

##Model
![model](https://github.com/cly1022/socialmedia-CNT/assets/17700771/f99bb531-45e0-47aa-8179-cd04a49ea056)


##Environment
Windows Server 2022 Standard

12th generation Intel(R) Core(TM) i9-12900K   3.2GHz

128 GB

NVIDIA RTX A6000

Python 3.7.6
pytorch==1.12.1
If you have any questions about the paper and repository, feel free to contact Leiyang Chen (cly_edu@whu.edu.cn) or open an issue!
