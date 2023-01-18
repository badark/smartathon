import os, sys
import torch
import torch.utils.data as data
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from argparse import ArgumentParser

from dataset import SmartathonImageDataset
from utils import init_model, init_optimizer, collate_fn

import logging


def save_checkpoint(model, optimizer, metrics, path):
    checkpoint_dict = {'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
    checkpoint_dict.update(metrics)
    torch.save(checkpoint_dict, path)


def main(args):
    model, transforms = init_model(args)
    img_dir = os.path.join(args.data_dir,'resized_images/')
    train_data = SmartathonImageDataset(os.path.join(args.data_dir, 'train_split.json'), img_dir, transform=transforms)
    val_data = SmartathonImageDataset(os.path.join(args.data_dir, 'val_split.json'), img_dir, transform=transforms)

    train_iterator = data.DataLoader(train_data, shuffle=True, 
        batch_size=args.batch_size, collate_fn=collate_fn)
    valid_iterator = data.DataLoader(val_data, shuffle=False,
        batch_size=args.valid_batch_size, collate_fn=collate_fn)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer, lr_scheduler = init_optimizer(model, args)
    meanAP = MeanAveragePrecision(iou_type="bbox")

    cpu_device = torch.device('cpu')
    device = torch.device('cuda')
    model.to(device)
    
    def train_step(engine, batch):
        model.train()
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'img_keys'} for t in targets]
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        lr_scheduler.step()
        return losses.item()

    trainer = Engine(train_step)
    timer = Timer()
    timer.attach(trainer, start=Events.STARTED, resume=Events.ITERATION_STARTED,
                pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(cpu_device) for k, v in t.items() if k != 'img_keys'} for t in targets]
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            meanAP.update(outputs, targets)

    evaluator = Engine(validation_step)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(trainer):
        logging.info(f"Epoch[{trainer.state.epoch}] Iteration[{trainer.state.iteration}] \
            LR{lr_scheduler.get_last_lr()} \
            Loss: {trainer.state.output:.2f} \
            Time: {timer.value():.2f}s")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_iterator)
        meanAP_metrics = meanAP.compute()
        logging.info(f"Validation Results - Epoch[{trainer.state.epoch}]  \
            meanAP: {meanAP_metrics['map']:.2f} - Time({trainer.state.times[Events.EPOCH_COMPLETED]:.2f}s)")
        logging.info("\tMetrics:" + ",".join([f"{k}: ({v:.3f})" for k, v in meanAP_metrics.items()])) 
        chkpt_path = os.path.join(args.output_prefix, f'checkpoint_{trainer.state.epoch}.pt')
        logging.info(f"Saving checkpoint to {chkpt_path}...")
        save_checkpoint(model, optimizer, meanAP_metrics, chkpt_path)
        meanAP.reset()

    trainer.run(train_iterator, max_epochs=10)

if __name__ == "__main__":
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model_type', dest='model_type', default="resnet50",
        help='type of model to train, see utils.py for supported model types')
    parser.add_argument('-op', '--optim_type', dest='optim_type', default="sgd",
        help='type of optimizer for training, see utils.py for supported optimizer types')
    parser.add_argument('-d', '--data_dir', dest='data_dir', 
        help='root directory of the dataset')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=8, type=int,
        help="batch size during training")
    parser.add_argument('-vb', '--valid_batch_size', dest='valid_batch_size', default=8, type=int,
        help="batch size during validation")
    parser.add_argument('-o', '--output_prefix', dest='output_prefix',
        help="output path for model checkpoints")
    parser.add_argument('--lr', dest='learning_rate', type=float, default=1e-5,
        help="learning rate for optimizer")
    args = parser.parse_args()
    os.makedirs(args.output_prefix, exist_ok = True) 
    logging.basicConfig(filename=os.path.join(args.output_prefix, 'log.txt'),level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    main(args)
