# MultiRobustBench: Benchmarking Robustness Against Multiple Attacks

**Leaderboard:** https://multirobustbench.github.io

## What is MultiRobustBench?
MultiRobustBench is a benchmark for evaluating progress in multiattack robustness.  While the bulk of research in adversarial ML focuses on robustness against specific perturbation types(mainly Lp bounded attacks), in practice, we would like our models to be robust against a wide variety of perturbations.  The goal of MultiRobustBench is to provide researchers with a standardized evaluation procedure for defenses against multiple attacks, allowing researchers to understand the current state of multiattack robustness and diagnose weaknesses of existing defenses (through performance visualizations available on the leaderboard site).

## Contributing to MultiRobustBench
We are always looking forward to contributions to MultiRobustBench in the form of new attacks and new defenses.  Please see the following sections on how to add new attacks and defenses.  We are also open to suggestions for evaluations or leaderboard; to suggest a new feature please open an issue with the Feature Suggestions template.

### Adding a new attack
To add a new attack to MultiRobustBench, please open an issue with the New Attack(s) template.  Additionally, please start a pull request which adds code for generating your attack in the attacks directory (and update the ```__init__.py``` file to import the attack).  The attack should be implemented as a class that inherits from nn.Module and calling the forward method with the inputs and labels should output an adversarial example for that input.  An example is shown below:
```python
class L1Attack(nn.Module):
    # use L1 APGD from http://proceedings.mlr.press/v139/croce21a/croce21a.pdf for training
    def __init__(self, model, bound=12, **kwargs):
        super().__init__()

        self.model = model
        self.attack = APGDAttack(model, norm='L1', eps=bound)

    def forward(self, inputs, labels):
        return self.attack.perturb(inputs, y=labels)
```

### Adding a new model
To add a new model to the leaderboard, please open an issue with the New Model(s) template.

# Evaluating models
The function ```evaluate_bin``` in ```eval.py``` is used to compute adversarial accuracies across
the set of attacks.  The attacks evaluated on are specified in ```cifar10_eval_config.yml```.  To evaluate a model M:

```python
from eval import evaluate_bin

evaluate_bin(M, "output", data_dir="data", batch_size=100)
```

```evaluate_bin``` will create 2 files: ```output_accs.json``` and ```output_accs.yml``` containing the individual adversarial accuracies with respect to each attack at each strength level.  To compute average and worst-case CR metrics:
```python
import json
from metrics import CR_ind_avg, CR_ind_worst, ATA, LEADERBOARD_SET
defense_accs = {}
with open("output_accs.json") as f:
    defense_accs = json.load(f)
cr_avg, cr_avg_single = CR_ind_avg(LEADERBOARD_SET, def_accs)
cr_worst, cr_worst_single = CR_ind_worst(LEADERBOARD_SET, def_accs)
```
The variable ```cr_avg``` stores the CR score for the average robustness leaderboard while ```cr_avg_single``` stores a dictionary of cr values measured across each individual attack type.  Similarily, ```cr_worst``` stores the CR score for the worst-case robustness leaderboard.