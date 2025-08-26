
# Simple ML
*Training neural networks with explicit, pythonic code.*

Most mainstream ML libraries abstract away many implementation details. 
While this simplifies model training, it can obscure the underlying mechanics 
and make it harder to understand or customize various required steps. 🧾

Therefore, in this repository we aim for **explicitness**,
e.g. in the [data pipeline](src/simple_ml/data_iterator.py). 
We also strive for a more pythonic way, e.g. in our implementation of the 
[training step](src/simple_ml/training_step.py). 🐍

For the training scripts, see [main.py](src/simple_ml/main.py)


# ⚙️ Installation:
```
pip install -e .
```


# 🛠️ Training:
From the `src` folder, run:
```
python simple_ml\main.py
```
