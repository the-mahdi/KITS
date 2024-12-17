# KITS: Inductive Spatio-Temporal Kriging with Increment Training Strategy

This repository contains the code for our AAAI 2025 [paper](https://arxiv.org/abs/2311.02565) "*KITS: Inductive Spatio-Temporal Kriging with Increment Training Strategy*", where we design an increment training strategy for inductive spatio-temporal kriging.

> **Abstract**: *Sensors are commonly deployed to perceive the environment. However, due to the high cost, sensors are usually sparsely deployed. Kriging is the tailored task to infer the unobserved nodes (without sensors) using the observed nodes (with sensors). The essence of kriging task is transferability. Recently, several inductive spatio-temporal kriging methods have been proposed based on graph neural networks, being trained  based on a graph built on top of observed nodes via pretext tasks such as masking nodes out and reconstructing them. However, the graph in training is inevitably much sparser than the graph in inference that includes all the observed and unobserved nodes. The learned pattern cannot be well generalized for inference, denoted as <i>graph gap</i>. To address this issue, we first present a novel <i>Increment</i> training strategy: instead of masking nodes (and reconstructing them), we add virtual nodes into the training graph so as to mitigate the graph gap issue naturally. Nevertheless, the empty-shell virtual nodes without labels could have inferior features and lack supervision signals. To solve these issues, we pair each virtual node with its most similar observed node and fuse their features together; to enhance the supervision signal, we construct reliable pseudo labels for virtual nodes. As a result, the learned pattern of virtual nodes could be safely transferred to real unobserved nodes for reliable kriging. We name our new <u>K</u>riging model with <u>I</u>ncrement <u>T</u>raining <u>S</u>trategy as KITS. Extensive experiments demonstrate that KITS consistently outperforms existing methods by large margins, e.g., the improvement over MAE score could be as high as 18.33%.*

## Dependencies

- Python 3.8
- PyTorch 1.8.1
- PyTorch Lightning 1.4.0
- cuda 11.1
```
> conda env create -f env_{ubuntu,windows}.yaml
```

## Datasets

We utilize 8 datasets from different field in this paper:
- Traffic speed datasets:
    - METR-LA
    - PEMS-BAY
    - SEA-LOOP
- Traffic flow dataset:
    - PEMS07
- Air quality datasets (PM2.5):
    - AQI36
    - AQI
- Solar power datasets:
    - NREL-AL
    - NREL-MD

These datasets could be downloaded from this [datasets.zip](https://drive.google.com/file/d/1VQrSLNAr3qr2LAsEK1-_CbBbu6vr0G63/view?usp=sharing), and compressed to the current path.

## Usage

- Run the following commands for training and testing.

- **Training**:
    - E.g., train KITS with missing ratio of 0.5:
      ```
      python train.py --config config/kits/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}.yaml --miss-rate 0.5 --lr 0.0002 --patience 50
      ```

- **Testing**
    - The pretrained KITS (with random seed 1, with missing ratio 0.5) could be downloaded from [final_model.zip](https://drive.google.com/file/d/1uj74MTy6zukWrnwQ_3zoMP67hurtnerP/view?usp=sharing).
    - E.g., test KITS with missing ratio of 0.5:
      ```
      python train.py --config config/kits/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}.yaml --pretrained-model final_model/{la_point, bay_point, sea_loop_point, pems07_point, aqi36, aqi, nrel_al_point, nrel_md_point}/final/seed1_rm0.5/best.ckpt
      ```

## References

This repo is mainly built based on [grin](https://github.com/Graph-Machine-Learning-Group/grin). Thanks for their great work!
