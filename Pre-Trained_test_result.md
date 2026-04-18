# Pre-trained EfficientNetB0

### Experiment Setup
Trained on filtered Cat and Dog images (image size larger than 16 KBs)

(5000 cats, 5000 Dogs)

Training Vs Validation: 80% : 20%

Trained for 5 Epochs, Validated for 3 Epochs

Data augmentation including (random flip, random rotation)

Test Set: 2500 cats, 2500 Dogs

<hr>

### result

result output:
```
(.venv) PS Dog_Cat_Classification_CNN> python .\infr_EfficientNet.py 
Using device: cpu
Classes: ['cats', 'dogs']
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Training classifier...

Epoch 1:
Train Loss: 0.2167, Acc: 0.9361
Val   Loss: 0.1315, Acc: 0.9611

Epoch 2:
Train Loss: 0.1254, Acc: 0.9555
Val   Loss: 0.1087, Acc: 0.9631

Epoch 3:
Train Loss: 0.1073, Acc: 0.9605
Val   Loss: 0.1000, Acc: 0.9649

Epoch 4:
Train Loss: 0.1041, Acc: 0.9608
Val   Loss: 0.0987, Acc: 0.9635

Epoch 5:
Train Loss: 0.0919, Acc: 0.9670
Val   Loss: 0.0919, Acc: 0.9624


Fine-tuning...

Epoch 1:
Train Loss: 0.0785, Acc: 0.9715
Val   Loss: 0.0597, Acc: 0.9779

Epoch 2:
Train Loss: 0.0528, Acc: 0.9822
Val   Loss: 0.0530, Acc: 0.9800

Epoch 3:
Train Loss: 0.0403, Acc: 0.9865
Val   Loss: 0.0456, Acc: 0.9817


Final evaluation on TEST set...

TEST Loss: 0.0549, TEST Acc: 0.9802

```
<hr>

### Inference Example

```
(.venv) PS \Dog_Cat_Classification_CNN> 
python .\use_trained_eff_net.py

Script started               
Using device: cpu
Image loaded
Running inference
Dog (1.0000)
```