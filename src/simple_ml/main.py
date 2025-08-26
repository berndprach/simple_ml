from functools import partial

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from simple_ml import cifar10, metrics, augmentation, simple_convolutional_network
from simple_ml.data_iterator import DataIterator
from simple_ml.training_step import general_training_step

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.1
NUMBER_OF_EPOCHS = 24
BATCH_SIZE = 256
MOMENTUM = 0.9
USE_NESTEROV_MOMENTUM = True
USE_TEST_DATA = False
CROSS_ENTROPY_TEMPERATURE = 8.
CROP_SIZE = 4


def main():
    training_data, evaluation_data = cifar10.load_data(use_test_data=USE_TEST_DATA)

    train_loader = DataIterator(training_data)
    train_loader.shuffle().repeat(NUMBER_OF_EPOCHS).batch(BATCH_SIZE, stack=True).to(DEVICE)
    train_loader.convert_x_to(torch.float32).rescale_x(1/255).center_x(cifar10.means)
    train_loader.apply_to_x(augmentation.crop_flip_erase(crop_size=CROP_SIZE))

    evaluation_loader = DataIterator(evaluation_data)
    evaluation_loader.batch(BATCH_SIZE, stack=True).to(DEVICE)
    evaluation_loader.convert_x_to(torch.float32).rescale_x(1/255).center_x(cifar10.means)

    model = simple_convolutional_network.load()
    model.to(DEVICE)

    optimizer = SGD(model.parameters(), lr=0., momentum=MOMENTUM, nesterov=USE_NESTEROV_MOMENTUM)
    training_step = partial(general_training_step,
        loss_function = partial(metrics.cross_entropy_with_temperature, temperature=CROSS_ENTROPY_TEMPERATURE),
        optimizer = optimizer,
        scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=len(train_loader))
    )

    print("Training:")
    for i, (x_batch, y_batch) in enumerate(train_loader, start=1):
        predictions = model(x_batch)
        training_step(predictions, y_batch)
        batch_accuracy = metrics.accuracy(predictions, y_batch).mean().item()
        print(f"Step {i: 4d}/{len(train_loader)}, Accuracy: {batch_accuracy:.1%}", end="\r")

    print("\n\nEvaluating:")
    model.eval()
    model = torch.inference_mode()(model)
    correct_count = 0, 0
    for x_batch, y_batch in evaluation_loader:
        predictions = model(x_batch)
        correct_count += int(metrics.accuracy(predictions, y_batch).sum().item())
    print(f"Validation Accuracy: {correct_count / len(evaluation_data):.1%}")


if __name__ == "__main__":
    main()