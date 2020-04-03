import cv2
from torchvision import models, transforms

class Imagetransform(object):
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(
                    resize,scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ]),
            'val' : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        }
    def __call__(self,img,phase='train'):

        return self.data_transform[phase](img)





class CIFARDataset(object):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.label = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,index):
        img = self.img_list[index]
        img_transformed = self.transform(img, self.phase)

        return img_transformed, int(self.label[index])

    
