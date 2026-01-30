import numpy as np
import tensorflow as tf

from datasets import PascalVOCDataset

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        print('dataset_path->',os.path.join(data_folder, self.split + '_images.json'))

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        #print('get_item')
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        #print('images_shape->',np.array(image).shape)

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        #print('image->',image)
        #print('objects[boxes]->',objects['boxes'])
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        #print('dataset_boxes->',boxes)
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

 # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)

def run_train(dataset, num_epochs=1):
    start_time = time.perf_counter()
    print('run_train')
    model = SSD(n_classes=20)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy)
    print('dataset->',dataset)

    for _ in tf.data.Dataset.range(num_epochs):
        for idx,(images,boxes,labels) in enumerate(dataset): # (batch_size (N), 300, 300, 3)
            print('=========================================================')
            images = np.array(images)
            labels = np.array(labels)
            boxes = np.array(list(boxes))
            print('image_shape->', images.shape)
            print('labels_shape->',labels.shape)
            print('boxes_shape->',boxes.shape)

            if isprint: tf.print(type(images), type(labels),images.shape,labels.shape)
            predicted_locs, predicted_socres = model(images)# (N, 8732, 4), (N, 8732, n_classes)
            loss = criterion(predicted_locs,predicted_socres,boxes,labels)
            print('loss->',loss)
            if idx ==10: break
        pass
    tf.print("실행 시간:", time.perf_counter() - start_time)
def train():
    print('train')
    print(tf.__version__)
    batch_size= 8
    images,boxes,labels,difficulties,new_boxes= PascalVOCDataset()
    new_boxes = list(new_boxes)


    boxes = tf.ragged.constant(boxes)
    labels = tf.ragged.constant(labels)
    new_boxes = tf.ragged.constant(new_boxes)

    print('boxes->',boxes.shape)
    print('labels->',labels.shape)
    print('images->', np.array(images).shape)

    dataset = tf.data.Dataset.from_tensor_slices((images,new_boxes,labels))
    run_train(dataset.map(resize_image_bbox, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))

def main():
    train()
if __name__ =='__main__':
    main()