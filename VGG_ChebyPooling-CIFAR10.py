import os
import torch
import torchvision
import pytorch_lightning

from VGG_ChebyPooling import VGG16_ChebyPooling

class Model(VGG16_ChebyPooling):
    def __init__(self):
        super().__init__(num_classes=10)
        print(self)

    def prepare_data(self):
        self.trainSet = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True)
        #print(self.trainSet.data.dtype, self.trainSet.data.shape) #uint8 (50000, 32, 32, 3)
        #print(len(self.trainSet)) #50000

        self.valSet = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True)
        #print(self.trainSet.data.dtype, self.trainSet.data.shape) #uint8 (50000, 32, 32, 3)
        #print(len(self.trainSet)) #50000

        self.testSet = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True)
        #print(self.testSet.data.dtype, self.testSet.data.shape) #uint8 (10000, 32, 32, 3)
        #print(len(self.testSet)) #10000

    def setup(self, stage):
        if (stage == 'fit'):
            self.trainSet.transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.5, std=0.5),
                torchvision.transforms.RandomErasing(),
            ])
            self.valSet.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.5, std=0.5),
            ])
            self.trainData, self.valData = torch.utils.data.random_split(self.trainSet, [40000, 10000])
            self.valData.dataset = self.valSet
        elif (stage == 'test'):
            self.testSet.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.5, std=0.5),
            ])
            self.testData = self.testSet

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainData, batch_size=100, num_workers=os.cpu_count(), shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valData, batch_size=100, num_workers=os.cpu_count())

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.testData, batch_size=100, num_workers=os.cpu_count())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'reduce_on_plateau': True, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        data, label = batch
        pred_label = self(data)
        return torch.nn.functional.cross_entropy(pred_label, label)
        
    def validation_step(self, batch, batch_idx):
        data, label = batch
        pred_label = self(data)
        loss = torch.nn.functional.cross_entropy(pred_label, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = torch.mean(pred_label.argmax(-1) == label, dtype=torch.float)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, label = batch
        pred_label = self(data)
        acc = torch.mean(pred_label.argmax(-1) == label, dtype=torch.float)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        print()

model = Model()
trainer = pytorch_lightning.Trainer(gpus=-1, max_epochs=300)
trainer.fit(model)
trainer.test(model)
