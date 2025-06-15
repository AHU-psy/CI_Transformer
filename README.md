# CI-Transformer-
Code and dataset for “Context_Interaction_Transformer_for_Insulator_Semantic_Segmentation_in_Infrared_Images”. 


---
We propose a semantic segmentation method for insulators in TL infrared images based on a Context Interaction Transformer (CI Transformer). The proposed method incorporates a novel Context-aware Transformer designed to learn diverse contextual information about insulators and achieve context information interaction across different receptive fields through multiple iterations. To adapt the model for semantic segmentation of various types of insulators in complex scenarios, the attention matrix of the CI Transformer calculates an edge loss to emphasize edge information during context interaction. Additionally, the integration of Group Instance Whitening Loss enhances the representational capacity of the backbone network.



---
Our code is based on the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) image segmentation framework. We integrate our custom-designed CI Transformer as a head module within the mmsegmentation framework to enable training and evaluation.



---


We release a dataset consists of 3,055 infrared images of transmission line insulators. The insulators in these images were annotated using the **Interactive Semi-Automatic Annotation Tool with Segment Anything**.
You can download the dataset from the following link:
[Download Dataset](https://drive.google.com/file/d/179GCvfT32noUsd2Uk7C0bLR3VmrM2NQY/view?usp=drive_link)

---

