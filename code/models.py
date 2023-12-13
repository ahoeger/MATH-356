import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedConvNet(nn.Module):
    def __init__(self):
        super(SimplifiedConvNet, self).__init__()

        # block 1
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # block 3 - additional depth
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Assuming the input size is (2, 128, 128), adjusting feature map size
        self.num_features = 64 * 16 * 16  # Adjusted after additional pooling layer

        # linear layers - Reduced the complexity
        self.fc1 = nn.Linear(self.num_features, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 62)  # Output classes

        self._initialize_weights()

    def forward(self, x):
        # block 1
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))

        # block 2
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))

        # block 3
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))

        # Flattening the output for the linear layers
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # linear layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)