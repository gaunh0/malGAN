# Malware generate using GAN
Base on *Generating Adversarial Malware Examples for Black-Box Attacks Based on GAN*

## Requirements
sklearn
pytorch
numpy
progress

## Running

### Command 
```
python main.py [-h] [--blackbox BLACKBOX] Z train_epoch retrain_epoch

positional arguments:
  Z                    Noise dimension
  train_epoch          Train epoch
  retrain_epoch        Retrain epoch

optional arguments:
  -h, --help           show this help message and exit
  --blackbox BLACKBOX  Blackbox model: RF: Random Forest LG: Logistic Regression DT: Decision
                       Tree (default) SVM: Support Vector Machine

```
### Example

```
python  main.py [-h] [--blackbox BLACKBOX] Z train_epoch retrain_epoch
```
