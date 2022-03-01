# CNM: Weakly Supervised Video Moment Localization with Contrastive Negative Sample Mining

In this work, we study the problem of video moment localization with natural language query and propose a novel weakly suervised solution by introducing Contrastive Negative sample Mining (CNM). 
Specifically, we use a learnable Gaussian mask to generate positive samples, highlighting the video frames most related to the query, and consider other frames of the video and the whole video as easy and hard negative samples respectively. We then train our network with the Intra-Video Contrastive loss to make our positive and negative samples more discriminative. 

Our paper was accepted by AAAI-2022. [[Paper](https://www.aaai.org/AAAI22Papers/AAAI-5056.ZhengM.pdf)] [[Project Page](https://minghangz.github.io/publication/cnm/)]

## Pipeline

![pipeline](imgs/pipeline.png)
