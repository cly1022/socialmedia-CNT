# Title

Identifying Influential Nodes In Complex Networks via Transformer

## Abstract

In the domain of complex networks, the identification of influential nodes plays a crucial role in ensuring network stability and facilitating efficient information dissemination. Although the study of influential nodes has been applied in many fields such as suppression of rumor spreading, regulation of group behavior, and prediction of mass events evolution, current deep learning-based algorithms have limited input features and are incapable of aggregating neighbor information of nodes, thus failing to adapt to complex networks.We propose an influential node identification method in complex networks based on the Transformer. In this method, the input sequence of a node includes information about the node itself and its neighbors, enabling the model to effectively aggregate node information to identify its influence. Experiments were conducted on 9 synthetic networks and 12 real networks. Using the SIR model and a benchmark method to verify the effectiveness of our approach.The experimental results show that this method can more effectively identify influential nodes in complex networks. In particular, the method improves 27 percent compared to the second place method in network Netscience and 21 percent in network Faa.

## Contributions

This paper brings several noteworthy contributions to the research field, which can be summarized as follows:

(1) In this paper, the Transformer model is introduced into the complex network to convert the node influence identification task into a regression task.

(2) By customizing the node sequence, the nodes can fuse the information of first-order neighbors and second-order neighbors. Experiments show that the Transformer can achieve the same aggregation effect as GNN, and the Transformer is more stable in complex networks.

(3) The CNT model is different from the previous GCN and GNN-based models in that the length of the input sequence can be dynamically adjusted according to the size of the complex network. Sequence features of different lengths can be input for various complex networks, which allows the CNT model to obtain enough node information so that it can handle complex networks of different sizes.

(4) We performed several parametric analyses to determine the optimal length of the input sequence as a way to obtain the most efficient model.

(5) The CNT model outperforms seven benchmark methods in identifying node influence across 9 synthetic and 12 real-world networks with various infection probabilities.

## Datesets

This repository includes the dataset introduced by the following paper: Adjnoun, Facebook, Netscience, Jazz, Lesmis, Hamster, Moreno, Plobooks, USAir PowerGrid etc. 
URL:[[https://networkrepository.com/index.php](https://networkrepository.com/index.php)]

## Model

![model](https://private-user-images.githubusercontent.com/17700771/328844541-f99bb531-45e0-47aa-8179-cd04a49ea056.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTUyNDg2NzMsIm5iZiI6MTcxNTI0ODM3MywicGF0aCI6Ii8xNzcwMDc3MS8zMjg4NDQ1NDEtZjk5YmI1MzEtNDVlMC00N2FhLTgxNzktY2QwNGE0OWVhMDU2LmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTA5VDA5NTI1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRhYThlYTc4YTBiOTQwNmJkZmVhYmNkMzU5MTBkMTM4Y2FhM2YzYWJkMWE3NzA1ZmQzOTQ1YTRlOWY1MzczOGEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.5GTdKrIzWIK26ue2cLevfYR3I7gwr8v8V3CGU_fgJvc)

## Environment

Windows Server 2022 Standard

12th generation Intel(R) Core(TM) i9-12900K 3.2GHz

128 GB

NVIDIA RTX A6000

Python 3.7.6 

pytorch==1.12.1


If you have any questions about the paper and repository, feel free to contact Leiyang Chen (cly_edu@whu.edu.cn) or open an issue!
