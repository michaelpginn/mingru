"""PyTorch (convolutional) MinGRU reference implementation

Christoph Heind, 2024
https://github.com/cheind/mingru

Based on:
    Were RNNs All We Needed?
    Leo Feng, 2024, https://arxiv.org/pdf/2410.01201v1
"""

import warnings
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


class ToVideo(torch.nn.Module):
    def forward(self, data):
        # Do some transformations
        return tv_tensors.Video(data)


def get_train_transform():
    return v2.Compose(
        [
            ToVideo(),
            v2.Resize((256, 256)),
            v2.RandomRotation(10),
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_test_transform():
    return v2.Compose(
        [
            ToVideo(),
            v2.Resize((224, 224)),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_dataset(cfg, train: bool, transform: Callable):
    ds = UCF101(
        cfg["ucf101_path"],
        cfg["ucf101_annpath"],
        frames_per_clip=15,
        step_between_clips=15,
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
            dropout=0.25,
            residual=True,
            bias=True,
        )

        self.conv_logits1 = torch.nn.Conv2d(cfg["hidden_sizes"][-1], 256, kernel_size=1)
        self.conv_logits2 = torch.nn.Conv2d(256, 101, kernel_size=1)

    def forward(self, video):
        B, S = video.shape[:2]
        features = self.backbone(video.flatten(0, 1)).unflatten(0, (B, S))
        # (B,S,512,28,28)
        out, _ = self.rnn.forward(features, h=None)
        # (B,S,256,2,2) with strides=2, padding=1
        out = self.conv_logits1(out[:, -1])
        out = F.relu(out)
        out = self.conv_logits2(out)
        # (B,101,2,2)
        return F.adaptive_max_pool2d(out, 1).squeeze(2, 3)


def train(cfg):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = UCF101Classifier(cfg).to(dev)
    # v = torch.randn(16, 10, 3, 224, 224).to(dev)

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
    sched = torch.optim.lr_scheduler.StepLR(
        optimizer, cfg["num_train_steps"] // 2, gamma=0.1
    )

    step = 0
    best_acc = 0.0
    while step < cfg["num_train_steps"]:
        for video, labels in dl_train:
            video = video.to(dev)
            labels = labels.to(dev)
            logits = classifier(video)
            loss = crit(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sched.step()
            correct = (logits.argmax(1) == labels).sum().item()
            accuracy = 100 * correct / len(logits)
            if step % 20 == 0:
                print(
                    f"Step {step+1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
                )  # NOQA
            if (step + 1) % 200 == 0:
                val_acc = evaluate(classifier, dev, dl_val)
                print(f"Step {step+1}, Validation Accuracy: {val_acc*100:.2f}%")
                if val_acc > best_acc:
                    scripted = torch.jit.script(classifier)
                    torch.jit.save(scripted, "ucf101_classifier_best.pt")
                    best_acc = val_acc
            if step >= cfg["num_train_steps"]:
                break
            step += 1

    return classifier


@torch.no_grad()
def evaluate(classifier: torch.nn.Module, dev: torch.device, dl):
    classifier.eval()

    total = 0
    total_correct = 0

    for step, (video, labels) in enumerate(dl):
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
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.eval()

    transform = get_test_transform()
    ds = get_dataset(cfg, train=False, transform=transform)
    indices = np.random.permutation(len(ds))
    ds_test = torch.utils.data.Subset(ds, indices[:100])
    dl_test = torch.utils.data.DataLoader(
        ds_test,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["dl_workers"],
        collate_fn=custom_collate,
    )

    test_acc = evaluate(classifier, dev, dl_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    import os

    cfg = {
        "ucf101_path": os.environ["UCF101_PATH"],
        "ucf101_annpath": os.environ["UCF101_ANNPATH"],
        "ucf101_fold": 1,
        "ucf101_workers": 10,
        "dl_workers": 4,
        "hidden_sizes": [64, 128, 256, 256],
        "num_train_steps": 20000,
        "num_test_steps": 100,
        "batch_size": 16,
        "lr": 1e-4,
    }
    model = train(cfg)

    # # Script and eval the model
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, "ucf101_classifier.pt")

    # scripted = torch.jit.load("ucf101_classifier.pt")
    test(cfg, scripted)
