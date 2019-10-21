import numpy as np
import json
import sys
import torch
from PIL import Image
from simple_net import SimpleNet
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])


class CifarDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.paths = [sample['image'] for sample in samples]
        self.labels = [sample['class_num'] for sample in samples]
        self.transform = transform
        self.channel_means = (0.4914, 0.4822, 0.4465)
        self.channel_stds = (0.2023, 0.1994, 0.2010)
        self.norm = transforms.Compose([
            transforms.Normalize(self.channel_means, self.channel_stds)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        raw_img = Image.open(self.paths[idx])
        if self.transform:
            raw_img = self.transform(raw_img)
        img = np.asarray(raw_img) / 255
        img = img.astype(np.float)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        img = self.norm(img)

        labels = np.array(self.labels[idx]).astype(np.double)
        return img, labels

def _get_dataset_loader(data, transform=None, shuffle=False):
    return torch.utils.data.DataLoader(
            CifarDataset(data, transform=transform), batch_size=64, shuffle=shuffle, num_workers=8)

def main(data_path, model_name):
    output_model = model_name + ".pth"
    with open(data_path) as f:
        data = json.load(f)

    np.random.seed(42)
    np.random.shuffle(data)
    training_data = [x for x in data if "train" in x["set"]]
    testing_data = [x for x in data if "test" in x["set"]]
    print(f"Size of training set: {len(training_data)}")
    print(f"Size of testing set: {len(testing_data)}")

    train_dataset_loader = _get_dataset_loader(training_data, transform=transform_train, shuffle=True)
    test_dataset_loader = _get_dataset_loader(testing_data)
    writer = SummaryWriter()

    net = SimpleNet(10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.1)
    
    print("training...")
    train_iter = 0
    for epoch in range(600):
        
        if epoch == 250 or epoch == 350 or epoch == 500:
            scheduler.step()
            print(f"LR Stepped to: {scheduler.get_lr()}")

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            outputs = net(inputs.float().cuda())
            loss = criterion(outputs, labels.long().cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i) % 100 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, i, running_loss / 100))
                writer.add_scalar("TrainLoss", (running_loss / 100), train_iter)
                train_iter += 1
                running_loss = 0.0

        # Test
        if epoch + 1 % 10:
            print("testing...")
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                i = 0.0
                for inputs, labels in test_dataset_loader:
                    outputs = net(inputs.float().cuda())
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.long().cuda()).sum().item()
                    i += 1
                test_accuracy = correct / total
            print(f"Test Accuracy: {test_accuracy}")
            writer.add_scalar("TestAccuracy", test_accuracy, epoch)
            print(f"Correct: {correct}, Incorrect: {total-correct}")
            # generate and plot ROC true-pos vs false-pos.
            # print what predicted looks like, create confusion matrix.
    
    # Save
    print("saving...")
    torch.save(net.state_dict(), output_model)


if __name__ == '__main__':
    data_path = sys.argv[1]
    main(data_path, sys.argv[2])

