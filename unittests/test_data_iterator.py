
from simple_ml import data_iterator

def test_shuffle():
    inputs = [1, 2, 3]

    outputs = list(data_iterator.shuffle(inputs))

    assert all(i in list(outputs) for i in inputs)


def test_repeat():
    inputs = [1, 2, 3]

    outputs = list(data_iterator.repeat(inputs, n=2))

    assert outputs == [1, 2, 3, 1, 2, 3]


def test_batch():
    inputs = [1, 2, 3, 4, 5, 6]

    outputs = list(data_iterator.batch(inputs, batch_size=2))

    assert outputs == [[1, 2], [3, 4], [5, 6]]


if __name__ == '__main__':
    test_functions = {name: obj for name, obj in globals().items() if name.startswith('test_')}
    print(test_functions)

    for name, test_function in test_functions.items():
        print(f"\nRunning {name}().")
        test_function()
        print(f"Success.")