import os
import torch
import torch.utils.data as data
from ignite.engine import Engine, Events
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from argparse import ArgumentParser

from dataset import SmartathonImageDataset
from model import init_model


def save_checkpoint(model, optimizer, metrics, path):
    checkpoint_dict = {'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}
    checkpoint_dict.update(metrics)
    torch.save(checkpoint_dict, path)


def main(args):
    model, transforms = init_model(args)
    img_dir = os.path.join(args.data_dir,'resized_images/')
    train_data = SmartathonImageDataset(os.path.join(args.data_dir, 'train_split.csv'), img_dir, transform=transforms)
    val_data = SmartathonImageDataset(os.path.join(args.data_dir, 'val_split.csv'), img_dir, transform=transforms)

    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    valid_iterator = data.DataLoader(val_data, batch_size=args.valid_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    meanAP = MeanAveragePrecision(iou_type="bbox")

    cpu_device = torch.device('cpu')
    device = torch.device('cuda')
    model.to(device)

    def train_step(engine, batch):
        model.train()
        images, targets = batch
        images = images.to(device)
        targets = {k: v.to(device) for k,v in targets.items() if k != 'img_paths'}
        images = list(image for image in images)
        labels = list(label.unsqueeze(0) for label in targets['labels'])
        boxes = [box.unsqueeze(0) for box in  targets['boxes']]
        targets = [dict(labels=label, boxes=box) for label, box in zip(labels, boxes)]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        lr_scheduler.step()
        return losses.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            images, targets = batch
            labels = list(label.unsqueeze(0) for label in targets['labels'])
            boxes = [box.unsqueeze(0) for box in  targets['boxes']]
            targets = [dict(labels=label, boxes=box) for label, box in zip(labels, boxes)]
            images = images.to(device)
            images = list(image for image in images)
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            meanAP.update(outputs, targets)

    evaluator = Engine(validation_step)

    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training_loss(trainer):
        print(f"Epoch[{trainer.state.epoch}] Iteration[{trainer.state.iteration}] Loss: {trainer.state.output:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        meanAP.reset()
        evaluator.run(valid_iterator)
        meanAP_metrics = meanAP.compute()
        print(f"Validation Results - Epoch[{trainer.state.epoch}]  \
            meanAP: {meanAP_metrics['map']:.2f} - Time({trainer.state.times[Events.EPOCH_COMPLETED]:.2f}s)")
        chkpt_path = os.path.join(args.output_prefix, f'checkpoint_{trainer.state.epoch}.pt')
        print(f"Saving checkpoint to {chkpt_path}...")
        save_checkpoint(model, optimizer, meanAP_metrics, chkpt_path)

    trainer.run(train_iterator, max_epochs=10)

if __name__ == "__main__":
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model_type', dest='model_type', 
        help='type of model to train, see model.py for supported model types')
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
    main(args)
