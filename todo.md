- [x] Evaluate baseline model (deberta-v3-large) after training a classifier head on frozen backbone
  ```python train.py --baseline```


- [x] CV for each LoRA rank with no handling of class imbalance (baseline)
  ```python train.py --ranks 8 16 32 64 --folds 2 --epochs 10 --batch-size 16 --lr 1e-4 --max-length 128 --balance none```

in rank_sweep_results.json

- [x] CV for each LoRA rank 32 with different handling of class imbalance

  weighted loss has not been effective - magnified gradients interacting badly with LoRA.
```python train.py --ranks 32 --folds 0 --balance oversample --lr 1e-4 --batch-size 16 --epochs 15```

- [x] Tune decision threshold
  ```python train.py --eval-adapter full_train_oversample```
No impact, consistent across thresholds

Re-do the above with official splits used properly

- [x] Train with masked entities
```python train.py --ranks 32 --folds 0 --balance oversample --lr 1e-4 --batch-size 16 --epochs 15 --masked```

- [x] Train with label sampling
  ```python train.py --label-smoothing 0.1```
 
- [ ] Train with classification head
  ```python train.py --ranks 32 --folds 0 --balance oversample --lr 1e-4 --batch-size 16 --epochs 30 --multi-head
python train.py --eval-adapter <adapter_dir> --multi-head```