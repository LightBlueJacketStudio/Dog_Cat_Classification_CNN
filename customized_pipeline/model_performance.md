tested on Apr.29

3 epoch, 1 test, 80:20 validation ratio

```
Classes: ['cats', 'dogs']
Using device: cpu
Epoch 1/3:
  Train - Loss: 0.6570, Acc: 0.6034
  Val   - Loss: 0.6211, Acc: 0.6451
  Saved new best model.

Epoch 2/3:
  Train - Loss: 0.6302, Acc: 0.6352
  Val   - Loss: 0.6334, Acc: 0.6571
  Saved new best model.

Epoch 3/3:
  Train - Loss: 0.6083, Acc: 0.6645
  Val   - Loss: 0.5608, Acc: 0.6975
  Saved new best model.


Final evaluation on test set...

Test Loss: 0.5815, Test Acc: 0.6830
```

6 epoch training on 10,000 data points:

```
Epoch 1/6:
  Train - Loss: 0.6590, Acc: 0.5988
  Val   - Loss: 0.6335, Acc: 0.6320
  Saved new best model.

Epoch 2/6:
  Train - Loss: 0.6284, Acc: 0.6363
  Val   - Loss: 0.5957, Acc: 0.6620
  Saved new best model.

Epoch 3/6:
  Train - Loss: 0.6127, Acc: 0.6649
  Val   - Loss: 0.5656, Acc: 0.6981
  Saved new best model.

Epoch 4/6:
  Train - Loss: 0.5879, Acc: 0.6908
  Val   - Loss: 0.5444, Acc: 0.7126
  Saved new best model.

Epoch 5/6:
  Train - Loss: 0.5622, Acc: 0.7136
  Val   - Loss: 0.6027, Acc: 0.6716

Epoch 6/6:
  Train - Loss: 0.5307, Acc: 0.7357
  Val   - Loss: 0.5093, Acc: 0.7426
  Saved new best model.

```