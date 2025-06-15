

````markdown
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

## 📌 Notes

* Please ensure you have installed MMSegmentation and its dependencies before training.
* For more details on configuration files and customization, refer to the [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/).

---

## 📧 Contact

For questions or issues, feel free to open an issue or contact the authors via the paper.

```

如果你还想添加实验结果展示（比如 Table、mIoU 图、可视化样例），或者是 `Installation` 部分，我也可以帮你扩展。
```
