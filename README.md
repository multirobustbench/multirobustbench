# Code for "MultiRobustBench: Benchmarking Robustness Against Multiple Attacks"

The function ```evaluate_bin``` in ```eval.py``` is used to compute adversarial accuracies across
the set of attacks.  The attacks evaluated on are specified in ```cifar10_eval_config.yml```.  To evaluate a model M:

```python
from eval import evaluate_bin

evaluate_bin(M, "output", data_dir="data", batch_size=100)
```

```evaluate_bin``` will create 2 files: ```output_accs.json``` and ```output_accs.yml``` containing the individual adversarial accuracies with respect to each attack at each strength level.  We can then add an entry into our leaderboard by editing the ```leaderboard_source/data/def_accs.js``` with an entry of the format:
```
'Defense_name': {
        'Details': {
            'Title': paper title,
            'Paper URL': link to  paper,
            'Architecture': architecture,
            'Attacks Used': {
                'Attack1': eps1,
                'Attack2': eps2
            },
            'Extra Data': true or false,
            'Comments': details about defense
        },
        'Accuracies': {values copied over from output_accs.json}
    }
```
Code for metric computation is located in ```leaderboard_source/scripts/utils.js```.  The code for $\text{CR}_\text{ind-avg}$ is in the ```CR_ind_avg``` function, $\text{CR}_\text{ind-worst}$ in ```CR_ind_worst```, and stability constant in ```SC```. There is also a python version of these functions in ```metrics.py```

For training models for both approximating $\text{acc}^*$ and results in Section 5.2 of the paper, we use ```adv_train.py``` with the set of attacks specified in ```cifar10_train_config.yml```.  The general command for running training is of the form 

```
python adv_train.py --arch resnet18 --normalize --data_dir data/cifar10 --model_dir trained_models --chkpt_iters 100 --attack ATTACK_NAME --eps EPSILON
```
Trained models are saved in trained_models.  From this directory, we keep models saved at the checkpoint with highest test accuracy (name of checkpoint ends in _model_best.pth) and evaluate them using ```get_adv_train_acc.py```

```
python get_adv_train_acc.py --data_dir data/cifar10 --at_ckpts_dir trained_models --out_file advtrain_accs --normalize --arch resnet18
```
This evaluates the adversarially trained models stored in trained_models for the attack and epsilon that they were trained with.