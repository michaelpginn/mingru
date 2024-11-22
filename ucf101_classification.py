"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
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
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
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


def get_dataset(cfg, train: bool, transform: Callable):
    ds = UCF101(
        cfg["ucf101_path"],
        cfg["ucf101_annpath"],
        frames_per_clip=10,
        step_between_clips=10,
        fold=cfg["ucf101_fold"],
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
            dropout=0.1,
            residual=True,
            bias=True,
        )

        self.fc1 = torch.nn.Linear(sum(cfg["hidden_sizes"]), 256)
        self.bc1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 101)

    def forward(self, video):
        B, S = video.shape[:2]
        features = self.backbone(video.flatten(0, 1)).unflatten(0, (B, S))
        # (B,S,512,28,28)
        out, hs = self.rnn.forward(features, h=None)

        hs = [F.adaptive_avg_pool2d(hh.squeeze(1), (1, 1)) for hh in hs]
        h = torch.cat(hs, 1).squeeze()
        y = self.fc1(h)
        y = F.relu(self.bc1(y))
        y = self.fc2(y)
        return y


def train(cfg):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = UCF101Classifier(cfg).to(dev)

    transform = get_train_transform()
    ds = get_dataset(cfg, train=True, transform=transform)

    indices = np.random.permutation(len(ds))
    ds_train = torch.utils.data.Subset(ds, indices[:-100])
    ds_val = torch.utils.data.Subset(ds, indices[-100:])

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
    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.StepLR(optimizer, cfg["num_epochs"] - 2, gamma=0.1)

    step = 0
    best_acc = 0.0
    for epoch in range(cfg["num_epochs"]):
        for video, labels in dl_train:
            video = video.to(dev)
            labels = labels.to(dev)
            logits = classifier(video)
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
            if (step + 1) % 200 == 0:
                val_acc = validate(classifier, dev, dl_val)
                _logger.info(
                    f"Epoch {epoch+1}, Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%"
                )
                if val_acc > best_acc:
                    scripted = torch.jit.script(classifier)
                    torch.jit.save(scripted, "ucf101_classifier_best.pt")
                    best_acc = val_acc
            step += 1
        sched.step()

    return classifier


@torch.no_grad()
def validate(classifier: torch.nn.Module, dev: torch.device, dl):
    classifier.eval()

    total = 0
    total_correct = 0

    for video, labels in dl:
        video = video.to(dev)
        labels = labels.to(dev)
        logits = classifier(video)

        total_correct += (logits.argmax(1) == labels).sum().item()
        total += len(video)

    acc = total_correct / total
    classifier.train()
    return acc


@torch.no_grad()
def test(cfg, classifier):
    # From each of these selected sub-volumes, we obtain 10 inputs for our model, i.e. 4
    # corners, 1 center, and their horizontal flipping. The final pre- diction score
    # is obtained by averaging across the sampled sub-volumes and their cropped regions.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()

    ds = get_dataset(cfg, train=False, transform=None)
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

            logits = classifier(input)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits, 0)
        pred = all_logits.mean(0)
        total_correct += pred.argmax(0).item() == label
        total += 1
        _logger.info(f"{vidx+1}/{n_videos}, acc {total_correct/total}")

    _logger.info(f"acc {total_correct/total}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler("ucf101.log.txt", mode="w"),
            logging.StreamHandler(),
        ],
    )

    import os

    cfg = {
        "ucf101_path": os.environ["UCF101_PATH"],
        "ucf101_annpath": os.environ["UCF101_ANNPATH"],
        "ucf101_fold": 1,
        "ucf101_workers": 10,
        "dl_workers": 4,
        "hidden_sizes": [64, 128, 256, 256],
        "num_epochs": 10,
        "batch_size": 16,
        "lr": 1e-3,
    }
    model = train(cfg)

    # # Script and eval the model
    # scripted = torch.jit.script(model)
    # torch.jit.save(scripted, "ucf101_classifier.pt")

    scripted = torch.jit.load("ucf101_classifier_best.pt")
    test(cfg, scripted)
