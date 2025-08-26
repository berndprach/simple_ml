
from simple_ml import data_iterator

def test_shuffle():
    inputs = [1, 2, 3]

    outputs = list(data_iterator.shuffle(inputs))

    assert all(i in list(outputs) for i in inputs)


def test_shuffle_immutable():
    inputs = (1, 2, 3)

    outputs = list(data_iterator.shuffle(inputs))

    assert all(i in list(outputs) for i in inputs)


def test_shuffle_generator():
    inputs = (i for i in [1, 2, 3])

    outputs = list(data_iterator.shuffle(inputs))

    assert all(i in list(outputs) for i in inputs)


def test_repeat():
    inputs = [1, 2, 3]

    outputs = list(data_iterator.repeat(inputs, n=2))

    assert outputs == [1, 2, 3, 1, 2, 3]


def test_repeat_generator():
    inputs = (i for i in [1, 2, 3])

    outputs = list(data_iterator.repeat(inputs, n=2))

    assert outputs == [1, 2, 3, 1, 2, 3], outputs


def test_batch():
    inputs = [1, 2, 3, 4, 5, 6]

    outputs = list(data_iterator.batch(inputs, batch_size=2))

    assert outputs == [[1, 2], [3, 4], [5, 6]]


def test_incomplete_batch():
    inputs = [1, 2, 3, 4, 5]

    outputs = list(data_iterator.batch(inputs, batch_size=2))

    assert outputs == [[1, 2], [3, 4], [5]], outputs



if __name__ == '__main__':
    test_functions = {name: obj for name, obj in globals().items() if name.startswith('test_')}
    print(f"Found {len(test_functions)} test functions.")

    import time
    start_time = time.time()

    for name, test_function in test_functions.items():
        print(f"\nRunning {name}().")
        test_function()
        print(f"Success.")

    execution_seconds = time.time() - start_time
    print(f"\nCompleted {len(test_functions)} test functions in {execution_seconds * 1_000:.1f} microseconds.")