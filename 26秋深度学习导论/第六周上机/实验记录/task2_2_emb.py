import os
import sys
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader

from model import SimpleCLIP, ContrastiveLoss, device
import torch.optim as optim
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MyFashionMNIST(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.label_dict = {
            0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
            4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker',
            8: 'Bag', 9: 'AnkleBoot'
        }
        self.vocab = {
            'a': 0, 'photo': 1, 'of': 2, 'T-shirt/top': 3,
            'Trouser': 4, 'Pullover': 5, 'Dress': 6, 'Coat': 7,
            'Sandal': 8, 'Shirt': 9, 'Sneaker': 10, 'Bag': 11,
            'AnkleBoot': 12
        }

    def generate_extra_attribute(self, label):
        label_caption = f"a photo of {self.label_dict[label]}"
        tokens = label_caption.split()
        token_ids = [self.vocab[token] for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, self.generate_extra_attribute(label)


def get_test_text(vocab, label_dict):
    text_10 = []
    for label in range(10):
        caption = f"a photo of {label_dict[label]}"
        tokens = caption.split()
        text_10.append([vocab[token] for token in tokens])
    return torch.tensor(text_10, dtype=torch.long)


def test_clip_model(clip_model, test_loader, text_10):
    clip_model.eval()
    correct_predictions = 0
    total_num = 0
    with torch.no_grad():
        for images, labels, texts in tqdm(test_loader):
            images, text_10 = images.to(device), text_10.to(device)
            image_features, text_features = clip_model(images, text_10)
            similarities = torch.matmul(image_features, text_features.T)
            best_match_idx = similarities.argmax(dim=1)
            correct_predictions += (best_match_idx == labels.to(device)).sum().item()
            total_num += images.shape[0]
        print(f"Accuracy: {correct_predictions / total_num * 100:.2f}%")


def main():
    emb_size = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('EMB_SIZE', '128'))
    print(f"emb_size: {emb_size}", flush=True)

    train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    train_dataset_new = MyFashionMNIST(train_dataset)
    test_dataset_new = MyFashionMNIST(test_dataset)

    loader_kwargs = dict(batch_size=32, num_workers=4, persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_dataset_new, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset_new, shuffle=False, **loader_kwargs)

    clip = SimpleCLIP(len(train_dataset_new.vocab), emb_size=emb_size).to(device)
    text_10 = get_test_text(train_dataset_new.vocab, train_dataset_new.label_dict)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(clip.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        clip.train()
        total_loss = 0.0
        for images, labels, texts in tqdm(train_loader):
            images, texts = images.to(device), texts.to(device)
            optimizer.zero_grad()
            image_features, text_features = clip(images, texts)
            loss = criterion(image_features, text_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}", flush=True)
        test_clip_model(clip, test_loader, text_10)


if __name__ == '__main__':
    main()
