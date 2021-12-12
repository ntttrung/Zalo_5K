# Training

Using python command written in train.sh file for training

```bash
sh train.sh
```

# Inference

```bash
sudo docker run --gpus all -v [path to test data]:/data -v [curent dỉ]:/results [images id/docker name] /bin/bash /model/predict.sh
```

Example

```bash
sudo docker run -v /home/administrator/Documents/Zalo_5K-main/data:/data -v /home/administrator/Documents/Zalo_5K-main/result:/result 87282a22222a bin/bash /model/predict.sh
```