import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dset, img_size=64):
        self.dset = dset
        self.tensor_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        return self.tensor_transform(self.dset[idx]["image"])


class DataModule(L.LightningDataModule):
    def __init__(self, img_size=64, batch_size=64):
        super().__init__()
        self.save_hyperparameters()

        ds = load_dataset(
            "KrorngAI/CelebA-faces"
        )  # this dataset contains column 'image' only
        ds = ds["train"].train_test_split(0.05, shuffle=True, seed=3407)

        self.train_set = CustomDataset(ds["train"])
        self.valid_set = CustomDataset(ds["test"])

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.hparams.batch_size, shuffle=False
        )


if __name__ == "__main__":
    dm = DataModule()
    print("now")
    batch = next(iter(dm.train_dataloader()))
    print(batch.shape)
