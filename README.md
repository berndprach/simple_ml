
# Simple ML
*Training neural networks with explicit, pythonic code.*

Most mainstream ML libraries abstract away many implementation details. 
While this simplifies model training, it can obscure the underlying mechanics 
and make it harder to understand or customize various required steps.

Therefore, in this repository we aim for **explicitness**,
e.g. in the [data pipeline](src/simple_ml/data_iterator.py).

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

# 📈 Results:
By making the data pipeline more explicit, we were able to easily modify the order of operations to enhance performance. Specifically, we load the images onto the GPU as integer tensors, and only convert them to floats afterwards. With this adapted pipeline, we matches the result of [SimpleConvNet](https://github.com/berndprach/SimpleConvNet), achieving 93.6% validation accuracy. The training process completes about 10% faster compared to the standard one using PyTorch's DataLoader.
