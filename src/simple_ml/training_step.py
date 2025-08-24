

def general_training_step(predictions, y_batch, *, loss_function, optimizer, scheduler):
    loss = loss_function(predictions, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scheduler.step()
