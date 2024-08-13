def train_batch_log(batch_idx:int, batch_size:int, dataset_length:int, loss:float):
    log_message = 'batch index: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
        batch_idx, batch_idx * batch_size, dataset_length, 100. * batch_idx * batch_size / dataset_length, 
        loss
    )
    print(log_message)
    
    
def valid_epoch_log(epoch:int, dataset_length:int, loss:float):
    log_message = '\n' + 'Valid Epoch: {} Avg. loss: {:.4f}'.format(epoch, loss)
    print(log_message + '\n\n')
    print('===============================================')