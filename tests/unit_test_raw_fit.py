import unittest
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from torchimize.optimizer.gna_opt import GNA


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)
 
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


class TorchimizerTest(unittest.TestCase):

    BATCH_SIZE = 50
    LR = 0.02/100
    EPOCH = 2

    def __init__(self, *args, **kwargs):
        super(TorchimizerTest, self).__init__(*args, **kwargs)

    def setUp(self):
 
        np.random.seed(666)
        X = np.linspace(-1, 1, 1000)
        y = np.power(X, 2) + 0.1 * np.random.normal(0, 1, X.size)
        print(X.shape)
        print(y.shape)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1024)
        self.X_train = torch.from_numpy(self.X_train).type(torch.FloatTensor)
        self.X_train = torch.unsqueeze(self.X_train, dim=1)
        self.y_train = torch.from_numpy(self.y_train).type(torch.FloatTensor)
        self.y_train = torch.unsqueeze(self.y_train, dim=1)
        self.X_test = torch.from_numpy(self.X_test).type(torch.FloatTensor)
        self.X_test = torch.unsqueeze(self.X_test, dim=1)

        self.torch_data = Data.TensorDataset(self.X_train, self.y_train)
        self.loader = Data.DataLoader(dataset=self.torch_data, batch_size=self.BATCH_SIZE, shuffle=True)

        self.net = Net()

        self.opt = torch.optim.LBFGS(self.net.parameters(), lr=self.LR)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.LR)

        self.loss_func = nn.MSELoss()
    
    def test_gna_optimizer(self):

        self.opt = GNA(self.net.parameters(), lr=self.LR, model=self.net)

        all_loss = {}
        for epoch in range(self.EPOCH):
            print('epoch: ', epoch)
            for batch_idx, (b_x, b_y) in enumerate(self.loader):
                pre = self.net(b_x)
                loss = self.loss_func(pre, b_y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step(b_x)
                all_loss[epoch+1] = loss
                print('batch: {}, loss: {}'.format(batch_idx, loss.detach().numpy().item()))

        torch.save(self.net.state_dict(), './result/raw_train_fit_model.pth')

        self.net.eval()
        predict = self.net(self.X_test)
        predict = predict.data.numpy()
        plt.scatter(self.X_test.numpy(), self.y_test, label='origin')
        plt.scatter(self.X_test.numpy(), predict, color='red', label='predict')
        plt.legend()
        plt.show()


    def test_all(self):
        self.test_gna_optimizer()


if __name__ == '__main__':
    unittest.main()