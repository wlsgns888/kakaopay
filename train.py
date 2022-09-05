import torch
from utils import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
from data.dataset import CustomDataset
import torch.nn as nn
from model.trainer import Trainer
import os
from config.config import *
from config.experiment_config import *

class RunTrain:
    def __init__(self, exp_name, exp_config=None):
        self.test_name = exp_name
        self.exp_config = exp_config

    def run(self):
        self.set_member_value(MODEL_CONFIG)
        if self.exp_config:
            self.set_member_value(self.exp_config)
        self.transforms = [
            HorizontalFlip(p=0.5),
            Rotate(30, p=0.5),
            GaussianBlur(p=0.2),
            GaussNoise(p=0.5),
            Blur(p=0.2),
            RGBShift(p=0.5),
            ToGray(p=0.3),
            ColorJitter(p=0.3),
            RandomBrightness(p=0.5),
        ]
        if 'transforms' not in self.__dict__:
            self.transforms=[]
        
        save_folder_path = os.path.join(EXP_PATH, self.test_name)
        if not os.path.isdir(save_folder_path):
            os.mkdir(save_folder_path)
        
        save_model_path = os.path.join(save_folder_path, SAVE_MODEL_PATH)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_data_loader, val_data_loader, _ = set_data_loader(DATA_JSON_PATH, self.batch_size, self.worker, self.transforms)
        model = get_model(self.nc, device, use_pretrain=PRETRAIN_MODEL_PATH)

        optimizer = torch.optim.SGD(model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
        if device.type == 'cuda':
            model = model.cuda()

        loss_func = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            loss_func=loss_func,
            device=device,
            epochs=self.epochs,
            lr=self.lr,
            lr_decay=self.lr_decay,
            gamma=self.gamma,
            nc=self.nc
        )
        

        train_loss, train_acc, train_f1 = [], [], []
        val_loss, val_acc, val_f1 = [], [], []
        lr = []
        best_f1 = 0.0
        best_acc = 0.0
        best_loss = 1.0
        for epoch in range(self.epochs):
            print(f'Epoch : {epoch + 1}')

            lr.append(trainer.optimizer.param_groups[0]['lr'])
            t_loss, t_acc, t_f1 = trainer.train(train_data_loader, epoch + 1)
            train_loss.append(t_loss)
            train_acc.append(t_acc)
            train_f1.append(t_f1)
            print(
                f"Train loss : {t_loss}, Train acc : {t_acc}, F1-score : {t_f1}"
            )

            v_loss, v_acc, v_f1 = trainer.validate(val_data_loader, epoch + 1)
            val_loss.append(v_loss)
            val_acc.append(v_acc)
            val_f1.append(v_f1)
            print(
                f"Val loss : {v_loss}, Val acc : {v_acc}, F1-score : {v_f1}"
            )

            save_loss_graph(save_folder_path, train_loss, val_loss)
            save_acc_graph(
                save_folder_path, train_acc, train_f1, val_acc, val_f1
            )
            save_lr_graph(save_folder_path, lr)

            if best_loss > v_loss:
                print(f'Save best model')
                best_loss = v_loss
                best_f1 = v_f1
                best_acc = v_acc
                trainer.save_model(save_path = save_model_path)
                trainer.save_confusion_matrix(save_folder_path)
        return {'F1_score':best_f1, 'Accuracy':best_acc, 'loss':best_loss}

    def set_member_value(self, data):
        for k,v in data.items():
            if type(v) == str:
                exec(f'self.{k} = "{v}"')
            else:
                exec(f'self.{k} = {str(v)}')

def set_data_loader(data_json, batch_size, num_worker, transform_list=[], input_size=112):

    with open(data_json,'r') as json_file:
        train_val_data = json.load(json_file)

    train_list = train_val_data['train']
    val_list = train_val_data['val']
    # train_list.extend(train_val_data['train_aug'])
    # val_list.extend(train_val_data['val_aug'])
    test_list = train_val_data['val']

    val_test_trasforms = Compose(get_transform(input_size))

    transform_list.extend(get_transform(input_size))
    transforms = Compose(transform_list)

    train_dataset = CustomDataset(train_list, transforms)
    val_dataset = CustomDataset(val_list, val_test_trasforms, mode='val')
    test_dataset = CustomDataset(test_list, val_test_trasforms, mode='test')

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=num_worker, shuffle=True)
    val_data_loader   = DataLoader(val_dataset, batch_size=batch_size,num_workers=num_worker, shuffle=False)
    test_data_loader   = DataLoader(test_dataset, batch_size=1,num_workers=num_worker, shuffle=False)
    return train_data_loader, val_data_loader, test_dataset


if __name__ == '__main__':
    train = RunTrain('drop_out_test2')
    train.run()

    ##run learning experiments
    # save_result = {}
    # exp = exp_transform #exp_learning 
    # for exp_name, parameters in exp.items():
    #     train = RunTrain(exp_name, exp_config=parameters)
    #     ret = train.run()
    #     save_result[exp_name] = ret

    # with open('exp_transform.json', 'w') as json_file:
    #     json.dump(save_result, json_file, indent=4)