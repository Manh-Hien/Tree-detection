import os
from datetime import datetime
from BPlanDataset import BPlanDataset, train_test_split
import torch
from tools.helper import get_transform
from torch.utils.data import DataLoader, Dataset
from tools.model import get_model_object_detection
from tools.utils import collate_fn
from pycocotools.coco import COCO
from tools.engine import train_one_epoch, evaluate
from io import StringIO 
import sys
from metric import metric

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def train(dataset_path):

    dataset = BPlanDataset(dataset_path, get_transform(train=False))

    dataset_train, dataset_test = train_test_split(
            dataset, 0.2
        )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class =  len(dataset.category_list) +1
    model = get_model_object_detection(num_class)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    # lr value changed from 0.005 to 0.0005
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # train it for X epochs
    num_epochs = 50
    

    filename = datetime.now().strftime('%d-%m-%y-%H_%M_model')
    model_dict = filename + '/'
    PATH = "/opt/data/team/hien/models/"
    path_to_model = PATH + filename
    path_to_file = path_to_model + '/' + filename
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    #if __name__ == '__main__':
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader_train,
                        device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()


        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        #save every 10 epochs, if possible
        if epoch % 10 == 0 and num_epochs >= 10 and epoch != num_epochs:
            #save model
            torch.save(model.state_dict(), path_to_file +'_e'+ str(epoch) + '.pth')

    metric(model, dataset_test, num_class, device)
    torch.save(model.state_dict(), path_to_file +'_e'+ str(num_epochs) + '.pth')
   
    
    return model, path_to_file

def get_model(root_dir):
    with Capturing() as output:
        model, path = train(root_dir)
    
    training = []
    evaluation = []
    metric = []
    with open(path + '.txt', 'w') as txt_file:

        for line in output:
            txt_file.write(line + '\n')
        
    with open (path+'.txt', 'r') as txt_file_read:
        
        for line2 in txt_file_read:
            if line2.startswith("Epoch:"):
                training.append(line2)
            if line2.startswith("Test:"):
                evaluation.append(line2)
            if line2.startswith(" Average"):
                metric.append(line2)
                
    get_metric(path, training, evaluation, metric)
    return model

def get_metric(model_dir, training, evaluation, metric):
    with open(model_dir + '_metrics.txt', 'w') as metric_file:

        if training != []:
            metric_file.write(training[-1])
        if evaluation != []:
            metric_file.write(evaluation[-1])
        for average_metric in metric:
            metric_file.write(average_metric)


def main():
    root_dir = '/opt/data/team/hien/data/raw'
    get_model(root_dir)
    
        
    