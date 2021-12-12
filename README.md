# Training

Using python command written in train.sh file for training

```bash
sh train.sh
```

# Inference

```bash
sudo docker run --gpus all -v []:/data -v []:/results [images id] /bin/bash /model/predict.sh
```

Example

```bash
sudo docker run -v /home/administrator/Documents/Zalo_5K-main/data:/data -v /home/administrator/Documents/Zalo_5K-main/result:/result 87282a22222a bin/bash /model/predict.sh
```