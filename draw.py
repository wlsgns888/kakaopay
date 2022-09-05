ret_dict = {
    "lr_0005_step": {
        "F1_score": 0.952,
        "Accuracy": 0.9532,
        "loss": 0.1374
    },
    "lr_00005_step": {
        "F1_score": 0.9118,
        "Accuracy": 0.9148,
        "loss": 0.1966
    },
    "lr_0005_cos": {
        "F1_score": 0.9454,
        "Accuracy": 0.9468,
        "loss": 0.1337
    },
    "lr_00005_cos": {
        "F1_score": 0.9174,
        "Accuracy": 0.9201,
        "loss": 0.1762
    },
    "lr_0005_linear": {
        "F1_score": 0.9422,
        "Accuracy": 0.9437,
        "loss": 0.1616
    },
    "lr_00005_linear": {
        "F1_score": 0.9235,
        "Accuracy": 0.9258,
        "loss": 0.1679
    }
}

import matplotlib.pyplot as plt
import os

def save_result_graph(data_dict):

    plt.figure(figsize=(10, 7))
    plt.grid()
    #plt.plot(train_loss, label="train_loss")
    #plt.plot(valid_loss, label="valid_loss")
    F1_score = []
    Accuracy = []
    x_ = list(data_dict.keys())
    for k,v in data_dict.items():
        F1_score.append(v['F1_score'])
        Accuracy.append(v['Accuracy'])

    plt.plot(x_,F1_score, 'g^', label='F1_score')
    plt.plot(x_,Accuracy,'bs', label='Accuracy')
    plt.legend()
    plt.xlabel("Exp parameter", labelpad=15, fontdict={'weight':'bold', 'size':14})
    plt.ylabel("Metric", labelpad=15, fontdict={'weight':'bold', 'size':14})
    plt.title("Learning Exp", fontsize=16, fontdict={'weight':'bold'})
    #save_path = os.path.join(save_folder_path, "loss.png")
    plt.savefig('Learning_exp.png')
    plt.close()

if __name__ == '__main__':
    save_result_graph(ret_dict)