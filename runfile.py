import argparse
import logging
import sys
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import ImageFile
from models import densenet, alexnet, googlenet, mnasnet, mobilenet, resnet, shufflenetv2, squeezenet, vgg
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='train a special classification network')
    parser.add_argument(
        '--model-name',
        dest='model_name',
        help='model name (/model)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--train-images',
        dest='train_images',
        help='train images folder(/path/to/train_image_folder)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--test-images',
        dest='test_images',
        help='test images folder(/path/to/test_image_folder)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='output dir (/path/to/output_dir)',
        default='None',
        type=str
    )
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        help='batchsize',
        default=32,
        type=int
    )
    parser.add_argument(
        '--learning-rate',
        dest='learning_rate',
        help='learining rate (default=0.001)',
        default=0.001,
        type=float
    )
    parser.add_argument(
        '--epoch',
        dest='epoch',
        help='epoch (default=20)',
        default=20,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate
    EPOCH = args.epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = torch.cuda.is_available()

    data_transforms = {

        transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    transform = transforms.Compose([
        transforms.Resize(size=(227, 227)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root='/home/kevin/song/efficient_densenet_pytorch-master/images/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 从文件夹中读取validation数据
    validation_dataset = torchvision.datasets.ImageFolder(
        root='/home/kevin/song/efficient_densenet_pytorch-master/images/test', transform=transform)
    print(validation_dataset.class_to_idx)

    test_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if args.model_name == "densenet":
        Net = densenet.DenseNet().to(device)
    if args.model_name == "alexnet":
        Net = alexnet.AlexNet().to(device)
    if args.model_name == "googlenet":
        Net = googlenet.GoogLeNet().to(device)
    if args.model_name == "mobilenet":
        Net = mobilenet.MobileNetV2().to(device)
    if args.model_name == "mnasnet":
        Net = mnasnet.mnasnet1_0().to(device)
    if args.model_name == "squeezenet":
        Net = squeezenet.SqueezeNet().to(device)
    if args.model_name == "resnet":
        Net = resnet.resnet50().to(device)
    if args.model_name == "vgg":
        Net = vgg.vgg19().to(device)
    if args.model_name == "shufflenetv2":
        Net = shufflenetv2.shufflenet_v2_x1_0().to(device)

    criterion = nn.CrossEntropyLoss()
    opti = torch.optim.Adam(Net.parameters(), lr=LR)

    if __name__ == '__main__':
        Accuracy_list = []
        Loss_list = []

        for epoch in range(EPOCH):
            sum_loss = 0.0
            correct1 = 0

            total1 = 0
            for i, (images, labels) in enumerate(train_loader):
                num_images = images.size(0)

                images = Variable(images.to(device))
                labels = Variable(labels.to(device))

                if args.model_name == 'googlenet':
                    out = Net(images)
                    out = out[0]
                else:
                    out = Net(images)
                _, predicted = torch.max(out.data, 1)

                total1 += labels.size(0)

                correct1 += (predicted == labels).sum().item()

                loss = criterion(out, labels)
                print(loss)
                opti.zero_grad()
                loss.backward()
                opti.step()

                # 每训练100个batch打印一次平均loss
                sum_loss += loss.item()
                if i % 10 == 9:
                    print('train loss [%d, %d] loss: %.03f'
                          % (epoch + 1, i + 1, sum_loss / 2000))
                    print("train acc %.03f" % (100.0 * correct1 / total1))
                    sum_loss = 0.0
            Accuracy_list.append(100.0 * correct1 / total1)
            print('accurary={}'.format(100.0 * correct1 / total1))
            Loss_list.append(loss.item())

        x1 = range(0, EPOCH)
        x2 = range(0, EPOCH)
        y1 = Accuracy_list
        y2 = Loss_list

        total_test = 0
        correct_test = 0
        for i, (images, labels) in enumerate(test_loader):
            start_time = time.time()
            print('time_start', start_time)
            num_images = images.size(0)
            print('num_images', num_images)
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            print("GroundTruth", labels)
            if args.model_name == 'googlenet':
                out = Net(images)[0]
                out = out[0]
            else:
                out = Net(images)
            _, predicted = torch.max(out.data, 1)
            print("predicted", predicted)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
            print('time_usage', (time.time() - start_time) / args.batch_size)
        print('total_test', total_test)
        print('correct_test', correct_test)
        print('accurary={}'.format(100.0 * correct_test / total_test))

        plt.subplot(2, 1, 1)
        plt.plot(x1, y1, 'o-')
        plt.title('Train accuracy vs. epoches')
        plt.ylabel('Train accuracy')
        plt.subplot(2, 1, 2)
        plt.plot(x2, y2, '.-')
        plt.xlabel('Train loss vs. epoches')
        plt.ylabel('Train loss')
        # plt.savefig("accuracy_epoch" + str(EPOCH) + ".png")
        plt.savefig(args.output_dir + '/' + 'accuracy_epoch' + str(EPOCH) + '.png')
        plt.show()
        torch.save(args.output_dir, args.model_name + '.pth')


if __name__ == '__main__':
    args = parse_args()
    main(args)
