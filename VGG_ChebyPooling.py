import torch
import pytorch_lightning

#####################################################################################

class ChebyPooling2d(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.AvgPool2d = torch.nn.AvgPool2d(kernel_size=kernel_size, count_include_pad=False)
        self.MaxPool2d = torch.nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, input):
        mean = self.AvgPool2d(input)
        variance = self.AvgPool2d(input**2) - mean**2
        t = torch.nn.functional.softplus(self.MaxPool2d(input))
        return torch.nan_to_num(torch.div(variance, variance + (t - mean)**2))
    
#####################################################################################

class VGG_ChebyPooling(pytorch_lightning.LightningModule):
    def __init__(self, in_features, num_classes):
        super().__init__()

        self.example_input_array = torch.zeros(100, in_features, 32, 32)

        self.classifier = torch.nn.Sequential(
            #torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(512, num_classes),
            )

    def forward(self, input):
        f1 = self.set1(input)
        f2 = self.set2(f1)
        f3 = self.set3(f2)
        f4 = self.set4(f3)
        f5 = self.set5(f4)
        return self.classifier(f5)

#####################################################################################

class VGG16_ChebyPooling(VGG_ChebyPooling):
    def __init__(self, in_features=3, num_classes=1000):
        super().__init__(in_features, num_classes)

        self.set1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(True),
            ChebyPooling2d(2),
            )

        self.set2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            ChebyPooling2d(2),
            )

        self.set3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU(True),
            ChebyPooling2d(2),
            )

        self.set4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
            ChebyPooling2d(2),
            )

        self.set5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1), torch.nn.BatchNorm2d(512), torch.nn.ReLU(True),
            ChebyPooling2d(2),
            )
