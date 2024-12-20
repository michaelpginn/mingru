"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru
"""

import warnings
import logging
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader
from torchvision.datasets import UCF101
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms import v2
from torchvision import tv_tensors

import mingru

warnings.filterwarnings("ignore")

_logger = logging.getLogger("ucf101")


class ToVideo(torch.nn.Module):
    def forward(self, data):
        # Do some transformations
        return tv_tensors.Video(data)


def get_train_transform():
    return v2.Compose(
        [
            ToVideo(),
            v2.RandomPerspective(distortion_scale=0.2),
            v2.RandomChoice(
                [v2.RandomResizedCrop(224, scale=(0.3, 1.0)), v2.RandomCrop(224)]
            ),
            v2.RandomChannelPermutation(),
            v2.ColorJitter(brightness=0.5, hue=0.3, saturation=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_test_transform():
    return v2.Compose(
        [
            ToVideo(),
            v2.FiveCrop(224),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_quick_test_transform():
    return v2.Compose(
        [
            ToVideo(),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_dataset(cfg, train: bool, fold: int, transform: Callable):
    ds = UCF101(
        cfg["ucf101_path"],
        cfg["ucf101_annpath"],
        frames_per_clip=10,
        step_between_clips=10,
        fold=fold,
        train=train,
        output_format="TCHW",
        num_workers=cfg["ucf101_workers"],
        transform=transform,
    )

    return ds


def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)


# fold1: 98% top1 on val, 65% on test split
class UCF101Classifier(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.rnn = mingru.MinConv2dGRU(
            input_size=256,
            hidden_sizes=cfg["hidden_sizes"],
            kernel_sizes=3,
            strides=2,
            paddings=1,
            dropout=cfg["dropout"],
            residual=True,
            bias=True,
        )

        self.fc = torch.nn.Linear(sum(cfg["hidden_sizes"]), 101)

    def forward(self, video, h: list[torch.Tensor] | None = None):
        B, S = video.shape[:2]
        features = self.backbone(video.flatten(0, 1)).unflatten(0, (B, S))
        out, h = self.rnn.forward(features, h=h)

        h_pooled = [F.adaptive_avg_pool2d(hh.squeeze(1), (1, 1)) for hh in h]
        h_pooled = torch.cat(h_pooled, 1).squeeze()
        y = self.fc(h_pooled)
        return y, h


def train(cfg):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = UCF101Classifier(cfg).to(dev)

    transform = get_train_transform()
    fold = cfg["ucf101_fold"]
    ds = get_dataset(cfg, train=True, fold=fold, transform=transform)

    indices = np.random.permutation(len(ds))
    ds_train = torch.utils.data.Subset(ds, indices[:-200])
    ds_val = torch.utils.data.Subset(ds, indices[-200:])

    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["dl_workers"],
        collate_fn=custom_collate,
    )
    dl_val = torch.utils.data.DataLoader(
        ds_val,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["dl_workers"],
        collate_fn=custom_collate,
    )
    crit = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=cfg["lr"], weight_decay=5e-5
    )
    sched = torch.optim.lr_scheduler.StepLR(optimizer, cfg["num_epochs"] - 2, gamma=0.1)

    step = 0
    best_acc = 0.0
    best_loss = 1e5
    for epoch in range(cfg["num_epochs"]):
        for video, labels in dl_train:
            video = video.to(dev)
            labels = labels.to(dev)
            logits, _ = classifier(video)
            loss = crit(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct = (logits.argmax(1) == labels).sum().item()
            accuracy = 100 * correct / len(logits)
            if step % 20 == 0:
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
                )
            if (step + 1) % 500 == 0:
                val_acc, val_loss = validate(classifier, dev, dl_val)
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%, Validation Loss: {val_loss:.2f}"
                )
                if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                    scripted = torch.jit.script(classifier)
                    torch.jit.save(scripted, "ucf101_classifier_best.pt")
                    _logger.info("New best model")
                    best_acc = val_acc
                    best_loss = val_loss
            step += 1
        sched.step()

    return classifier


@torch.no_grad()
def validate(classifier: torch.nn.Module, dev: torch.device, dl, verbose: bool = False):
    classifier.eval()

    total = 0
    total_correct = 0
    total_loss = 0
    crit = torch.nn.CrossEntropyLoss()

    for video, labels in dl:
        video = video.to(dev)
        labels = labels.to(dev)
        logits, _ = classifier(video)
        loss = crit(logits, labels)

        total_correct += (logits.argmax(1) == labels).sum().item()
        total += len(video)
        total_loss += loss.item()
        if verbose:
            _logger.info(f"Acc: {total_correct/total}")

    acc = total_correct / total
    avg_loss = total_loss / total
    classifier.train()
    return acc, avg_loss


@torch.no_grad()
def test(cfg, classifier):
    # From each of these selected sub-volumes, we obtain 10 inputs for our model, i.e. 4
    # corners, 1 center, and their horizontal flipping. The final pre- diction score
    # is obtained by averaging across the sampled sub-volumes and their cropped regions.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()

    ds = get_dataset(cfg, train=False, transform=None, fold=cfg["ucf101_fold"])
    trans = get_test_transform()

    total = 0
    total_correct = 0

    # For each video
    labels = [ds.samples[i][1] for i in ds.indices]
    n_videos = len(ds.video_clips.video_paths)
    for vidx, label in zip(range(n_videos), labels):
        clips = ds.video_clips.subset([vidx])
        # randomly select up to 25 clips
        cids = np.random.permutation(len(clips))[:25]
        all_logits = []
        for cidx in cids:
            video, _, _, _ = clips.get_clip(cidx)

            # Five-crops + flip
            crops = trans(video)
            crops = torch.stack(crops, 0)
            hcrops = v2.functional.horizontal_flip(crops)
            input = torch.cat((crops, hcrops), 0).to(dev)

            logits, _ = classifier(input)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, 0)
        pred = all_logits.argmax(-1)
        unique, counts = torch.unique(pred, return_counts=True)
        majority_vote = unique[counts.argmax()]
        # pred = all_logits.mean(0)
        total_correct += majority_vote.item() == label
        total += 1
        _logger.info(f"{vidx+1}/{n_videos}, acc {total_correct/total:.2f}")

    _logger.info(f"test acc {total_correct/total:.2f}")


@torch.no_grad()
def quick_test(cfg, classifier):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()

    trans = get_quick_test_transform()
    ds = get_dataset(cfg, train=False, transform=trans, fold=cfg["ucf101_fold"])
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["dl_workers"],
        collate_fn=custom_collate,
    )
    acc, loss = validate(classifier, dev, dl, verbose=True)


if __name__ == "__main__":

    import os
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler("ucf101.log.txt", mode="a"),
            logging.StreamHandler(),
        ],
    )

    cfg = {
        "ucf101_path": os.environ["UCF101_PATH"],
        "ucf101_annpath": os.environ["UCF101_ANNPATH"],
        "ucf101_fold": 1,
        "ucf101_workers": 10,
        "dl_workers": 4,
        "hidden_sizes": [64, 128, 256, 256, 512],
        "dropout": 0.15,
        "num_epochs": 7,
        "batch_size": 16,
        "lr": 1e-4,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fold", type=int, choices=[1, 2, 3], required=True)
    subparsers = parser.add_subparsers(dest="cmd")
    train_parser = subparsers.add_parser("train", help="train")
    test_parser = subparsers.add_parser("test", help="test")
    test_parser.add_argument("--ckpt", default="ucf101_classifier_best.pt")
    test_parser.add_argument("--quick-test", action="store_true", default=False)
    args = parser.parse_args()

    cfg["ucf101_fold"] = args.fold

    if args.cmd == "train":
        _logger.info(f"New training session with {cfg}")
        classifier = train(cfg)
        quick_test(cfg, classifier)

    elif args.cmd == "test":
        _logger.info(f"New testing session with {cfg}")
        scripted = torch.jit.load(args.ckpt)
        cfg["quick_test"] = args.quick_test
        if args.quick_test:
            quick_test(cfg, scripted)
        else:
            test(cfg, scripted)
