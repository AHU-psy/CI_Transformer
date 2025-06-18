

# CI-Transformer

Code and dataset for the paper:  
**“Context Interaction Transformer for Insulator Semantic Segmentation in Infrared Images”**

---

## 🧠 Contribution

We propose a semantic segmentation method for insulators in transmission line (TL) infrared images, based on a novel **Context Interaction Transformer (CI Transformer)**.

- The CI Transformer introduces a **Context-aware Transformer** designed to learn diverse contextual features of insulators.
- It enables **context information interaction across different receptive fields** through iterative updates.
- To enhance performance in complex scenarios, the **attention matrix** incorporates an **edge loss**, emphasizing boundary information.
- A **Group Instance Whitening Loss** is also integrated to improve the backbone network’s representational capacity.

---


## 🔧 Installation

Before training, install the required dependencies:

- [MMSegmentation core dependencies](https://github.com/open-mmlab/mmsegmentation](https://mmsegmentation.readthedocs.io/zh-cn/latest/get_started.html)


-Specialized libraries
pip install pywt     # For wavelet convolution
pip install einops   # For rearrange operations

---

## 🚀 Training and Evaluation

This project is built upon the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework. We integrate our CI Transformer as a custom head module for semantic segmentation tasks.

To train the model with your configuration file, run:

```bash
python tools/train.py configs/ourcode/xxx.py
````

Replace `xxx.py` with the name of your specific configuration file.

---

## 📁 Dataset

We provide a dataset containing **3,055 infrared images** of transmission line insulators. All images were annotated using the
**Interactive Semi-Automatic Annotation Tool with Segment Anything**.

📥 [Download Dataset](https://drive.google.com/file/d/179GCvfT32noUsd2Uk7C0bLR3VmrM2NQY/view?usp=drive_link)

---

## 🛠️ Notes

* Please ensure you have installed MMSegmentation and its dependencies before training.
* For environment setup and installation instructions, refer to the [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/).
* Configuration files are located in `configs/ourcode/` and should be customized based on your experimental needs.

---

