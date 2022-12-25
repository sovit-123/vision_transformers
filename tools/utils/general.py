import torch
import matplotlib.pyplot as plt
import os

plt.style.use('ggplot')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is lower than the previous lowest, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, 
        model, 
        current_valid_loss, 
        epoch, 
        out_dir,
        # config,
        # model_name
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nLOWEST VALIDATION LOSS: {self.best_valid_loss}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                # 'config': config,
                # 'model_name': model_name
                }, f"{out_dir}/best_model.pth")

def save_model(out_dir, epoch, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir, 'last_model.pth'))

def save_loss_plot(
    out_dir, 
    train_loss_list,
    valid_loss_list, 
    x_label='epochs',
    y_label='train loss',
    save_name='loss'
):
    """
    Function to save both train loss graph.
    
    :param out_dir: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure_1.add_subplot()
    ax.plot(train_loss_list, color='tab:blue', label='train loss')
    ax.plot(valid_loss_list, color='tab:red', label='valid loss')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    figure_1.savefig(f"{out_dir}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')
    # plt.close('all')

def save_accuracy_plot(OUT_DIR, train_acc, valid_acc):
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.
    :param OUT_DIR: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        train_acc, color='tab:blue', label='train acc'
    )
    ax.plot(
        valid_acc, color='tab:red', label='valid acc'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('accuracy')
    ax.legend()
    figure.savefig(f"{OUT_DIR}/accuracy.png")
    # plt.close('all')

def set_training_dir(dir_name=None):
    """
    This functions counts the number of training directories already present
    and creates a new one in `runs/training/`. 
    And returns the directory path.
    """
    if not os.path.exists('runs/training'):
        os.makedirs('runs/training')
    if dir_name:
        new_dir_name = f"runs/training/{dir_name}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name
    else:
        num_train_dirs_present = len(os.listdir('runs/training/'))
        next_dir_num = num_train_dirs_present + 1
        new_dir_name = f"runs/training/res_{next_dir_num}"
        os.makedirs(new_dir_name, exist_ok=True)
        return new_dir_name