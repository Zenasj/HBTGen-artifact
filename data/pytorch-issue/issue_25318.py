import torch
import torchvision

class MyDataSet(Dataset):
    def __init__(self, img_list, transform):
        self.transform = transform
        self.imgs = []
        for img in img_list:
            self.imgs.append(img)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transform = transforms.Compose([transforms.ToTensor()])

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)

    videopath = 'intersection.mp4'
    vid = cv2.VideoCapture(videopath)
    success, frame = vid.read()
    image_array = []
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_array.append(frame)
        success, frame = vid.read()

    dataset = MyDataSet(image_array, transform)
    indices = list(range(len(dataset)))
    dataset_subset = torch.utils.data.Subset(dataset, indices[:])
    dataset_loader = torch.utils.data.DataLoader(dataset_subset, batch_size=1, shuffle=False, num_workers=4)

    tensor_imgs = []
    for i, img in enumerate(dataset_loader):
        tensor_imgs.append(img)