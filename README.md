# RL-Atari-with-Bayesian

This project is a Tensorflow version of Atari game reinforcement learner.
It also applies Bayesian Optimization via Hyperopt library to learn the better hyper-parameters tuple.

## How to run locally

clone repo

```
git clone https://github.com/ColinChen94/RL-Atari-with-Bayesian.git
```

create python3 virtual env

```
python3 -m venv env
. env/bin/activate
```

install requirements
```
pip install -r requirements.txt
```

go to main directory and install the module
```
pip install .
```

run the module with default parameters
```
python3 -m pong_trainer.task
```

## Preliminary Results

The result so far is here.
The best `(discount_factor, learning_rate, batch_size)` tuple is: `(0.980, 6.44e-05, 32)`.

![All returns](/docs/returns.png)