- [ ] Evaluate baseline model (deberta-v3-large) after training a classifier head on frozen backbone
  ```python train.py --baseline```
- [ ] CV for each LoRA rank with no handling of class imbalance (baseline)
  ```python train.py --ranks 8 16 32 64 --folds 2 --epochs 10 --batch-size 16 --lr 1e-4 --max-length 128 --balance none```
- [ ] CV for each LoRA rank with different handling of class imbalance
  ```python train.py --ranks 8 16 32 64 --folds 2 --epochs 10 --batch-size 16 --lr 1e-4 --max-length 128 --balance oversample```
  ```python train.py --ranks 8 16 32 64 --folds 2 --epochs 10 --batch-size 16 --lr 1e-4 --max-length 128 --balance weight```
  ```python train.py --ranks 8 16 32 64 --folds 2 --epochs 10 --batch-size 16 --lr 1e-4 --max-length 128 --balance both```
- [ ] Train best LoRA rank with back-translated augmentations
  ```python train.py --ranks 16 --folds 0 --epochs 10 --batch-size 16 --lr 1e-4 --max-length 128 --balance oversample --masked```