import torch
from torchvision import datasets, transforms, models
from tqdm import tqdm
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for feeding data through model.")
parser.add_argument("--dataset_name", type=str, choices=("dhmc_cd", "dhmc_luad", "dhmc_rcc"), help="Dataset to run evaluation on.")
parser.add_argument("--ckpt_file", type=Path, default=None, help="Location of saved model weights.")
parser.add_argument("--sd_prefix", type=str, default="", help="Prefix to remove from saved model weight keys.")

hparams = parser.parse_args()


# Find the mean and std.
if hparams.dataset_name == "dhmc_cd":
    hparams.data_path = Path("/workspace/jdipalma/DHMC_Data/CD/test")
    # Per-channel mean and standard deviation.
    hparams.mean = [0.8655948638916016, 0.8134953379631042, 0.8512848615646362]
    hparams.std = [0.1588381975889206, 0.2300935685634613, 0.18215394020080566]
    # Classes: Abnormal, Normal, Sprue
    hparams.class_names = ["Abnormal", "Normal", "Sprue"]
    hparams.num_classes = len(hparams.class_names)
elif hparams.dataset_name == "dhmc_rcc":
    hparams.data_path = Path("/workspace/jdipalma/DHMC_Data/RCC/test")
    # Per-channel mean and standard deviation.
    hparams.mean = [0.7893273234367371, 0.6810887455940247, 0.7918950915336609]
    hparams.std = [0.19146229326725006, 0.24683408439159393, 0.1701047122478485]
    # Classes: Benign, Chromophobe, Clearcell, Oncocytoma, Papillary
    hparams.class_names = ["Benign", "Chromophobe", "Clearcell", "Oncocytoma", "Papillary"]
    hparams.num_classes = len(hparams.class_names)
elif hparams.dataset_name == "dhmc_luad":
    hparams.data_path = Path("/workspace/jdipalma/DHMC_Data/LUAD/test")
    # Per-channel mean and standard deviation.
    hparams.mean = [0.8261354565620422, 0.739959180355072, 0.8314031362533569]
    hparams.std = [0.17794468998908997, 0.2484838217496872, 0.15636992454528809]
    # Classes: Acinar, Lepidic, Micropapillary, Papillary, Solid
    hparams.class_names = ["Acinar", "Lepidic", "Micropapillary", "Papillary", "Solid"]
    hparams.num_classes = len(hparams.class_names)
else:
    raise NotImplementedError


# Load the saved weights.
ckpt = torch.load(hparams.ckpt_file)
model = models.resnet18(num_classes=hparams.num_classes)
sd = {}
for k, v in ckpt["state_dict"].items():
    if k.startswith(hparams.sd_prefix):
        sd[k.replace(hparams.sd_prefix, "")] = v
sd["fc.weight"] = ckpt["state_dict"]["classifier.0.weight"]
sd["fc.bias"] = ckpt["state_dict"]["classifier.0.bias"]
print(model.load_state_dict(sd, strict=True))
for p in model.parameters():
    p.requires_grad = False
model = model.eval()
model = model.cuda()


ds = datasets.ImageFolder(hparams.data_path, 
                          transform=transforms.Compose([transforms.ToTensor(), 
                                                        transforms.Normalize(mean=hparams.mean, std=hparams.std)]))
dl = torch.utils.data.DataLoader(ds, batch_size=hparams.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)


preds = torch.empty(len(ds), hparams.num_classes, dtype=torch.float32)
labels = torch.empty(len(ds), dtype=torch.int64)


with torch.no_grad():
    for idx, batch in enumerate(tqdm(dl)):
        start = idx * hparams.batch_size
        end = start + hparams.batch_size

        preds[start:end] = model(batch[0].cuda(non_blocking=True))
        labels[start:end] = batch[1]

print(f"Accuracy:\t{(preds.max(dim=1)[1] == labels).float().mean():.4f}")
print(f"Scikit-Learn Accuracy:\t{accuracy_score(y_true=labels.numpy(), y_pred=preds.max(dim=1)[1].numpy()):.4f}")
print(f"F1-score:\t{f1_score(y_true=labels.numpy(), y_pred=preds.max(dim=1)[1].numpy(), average='macro'):.4f}")
print(f"AUC:\t{roc_auc_score(y_true=labels.numpy(), y_score=torch.nn.functional.softmax(preds, dim=1).numpy(), multi_class='ovr'):.4f}")
print(f"Classificaton Report:\t{classification_report(y_true=labels.numpy(), y_pred=preds.max(dim=1)[1].numpy(), digits=4, target_names=hparams.class_names)}")

