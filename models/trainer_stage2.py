import os

import numpy as np
from torch.optim import lr_scheduler
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from models.network.net_stage2 import net_stage2


class Trainer_stage2:
    def __init__(self, opt):
        self.model = net_stage2(opt)

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Total model parameters: {total_params:.2f}M")

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        trainable_ratio = (trainable_params / total_params) * 100
        print(f"Trainable parameters: {trainable_params:.2f}M ({trainable_ratio:.2f}%)")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.stage2_learning_rate, betas=(0.9, 0.999))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=opt.stage2_lr_decay_step, gamma=opt.stage2_lr_decay_factor)
        self.scaler = GradScaler()

        self.best_val_loss = float('inf')

    def train_epoch(self, dataloader: DataLoader, criterion):
        total_loss = 0.0
        total_batches = 0

        running_loss = 0.0
        batch_number = 0

        self.model.to(self.device)
        self.model.train()

        for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                output = self.model(data)
                loss = criterion(output.squeeze(1), target.type(torch.float32))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            total_loss += loss.item()
            batch_number += 1
            total_batches += 1

        return total_loss / (total_batches + 1)

    def validate_epoch(self, dataloader: DataLoader, criterion, epoch: int, writer: SummaryWriter = None):
        self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        dataset_preds = []
        dataset_targets = []

        for data, target in tqdm(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                with autocast():
                    pre = self.model(data)
                    loss = criterion(pre.squeeze(1), target.type(torch.float32))
                    running_loss += loss.item()
                    pre_prob = pre.cpu().numpy()
                    target = target.cpu().numpy()
                    dataset_preds.append(pre_prob)
                    dataset_targets.append(target)

        dataset_preds = np.concatenate(dataset_preds)
        dataset_targets = np.concatenate(dataset_targets)

        acc = accuracy_score(dataset_targets, dataset_preds > 0)
        ap = average_precision_score(dataset_targets, dataset_preds)

        if writer is not None:
            writer.add_scalar('Loss/Validation', running_loss / len(dataloader), epoch)
            writer.add_scalar('Accuracy', acc, epoch)
            writer.add_scalar('Average Precision', ap, epoch)

        return running_loss / len(dataloader), acc, ap

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, criterion, num_epochs: int,
              checkpoint_dir: str = None, writer: SummaryWriter = None):
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"Training" + "-" * 60)
            time.sleep(1)
            train_loss = self.train_epoch(train_dataloader, criterion)
            print(f"Validating" + "*" * 60)
            time.sleep(1)
            val_loss, acc, ap = self.validate_epoch(val_dataloader, criterion, epoch, writer=writer)

            print(
                f'{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}\nTrain Epoch: {epoch+1}: \n'
                f'train loss: {train_loss}\nval_loss:{val_loss}\nacc:{acc}\nap:{ap}')

            os.makedirs(checkpoint_dir, exist_ok=True)

            if (epoch+1) % 1000 == 0:
                checkpoint_path_1 = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path_1)
                print(f'Model checkpoint saved to {checkpoint_path_1}')

            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss and checkpoint_dir is not None:
                best_val_loss = val_loss
                checkpoint_path_2 = os.path.join(checkpoint_dir, f'model_best_val_loss.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, checkpoint_path_2)
                print(f'Model checkpoint saved to {checkpoint_path_2}')

            self.scheduler.step()

        print('Training complete.')
