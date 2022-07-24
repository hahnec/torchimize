__author__ = "Christopher Hahne"
__email__ = "inbox@christopherhahne.de"
__license__ = """
    Copyright (c) 2022 Christopher Hahne <inbox@christopherhahne.de>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from pathlib import Path

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

        self.plt_opt = False

    def setUp(self):
        
        torch.manual_seed(3008)

        X = torch.linspace(-1, 1, 1000)
        y = X**2 + 0.1 * torch.normal(0, 1, size=X.size())
        print(X.shape)
        print(y.shape)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.3, random_state=1024)
        self.X_train = torch.from_numpy(self.X_train).type(torch.FloatTensor)
        self.X_train = torch.unsqueeze(self.X_train, dim=1)
        self.y_train = torch.from_numpy(self.y_train).type(torch.FloatTensor)
        self.y_train = torch.unsqueeze(self.y_train, dim=1)
        self.X_test = torch.from_numpy(self.X_test).type(torch.FloatTensor)
        self.X_test = torch.unsqueeze(self.X_test, dim=1)

        self.torch_data = Data.TensorDataset(self.X_train, self.y_train)
        self.loader = Data.DataLoader(dataset=self.torch_data, batch_size=self.BATCH_SIZE, shuffle=True)

        self.net = Net()

        self.loss_func = nn.MSELoss()

        self.opt = GNA(self.net.parameters(), lr=self.LR, model=self.net)

    def test_optimizer(self):
        
        # for LBFGS usage
        def closure():
            pre = self.net(b_x)
            loss = self.loss_func(pre, b_y)
            self.opt.zero_grad()
            loss.backward()

            return loss

        all_loss = {}
        for epoch in range(self.EPOCH):
            print('epoch: ', epoch)
            for batch_idx, (b_x, b_y) in enumerate(self.loader):
                pre = self.net(b_x)
                loss = self.loss_func(pre, b_y)
                self.opt.zero_grad()
                loss.backward()

                # parameter update step based on optimizer
                if str(self.opt).__contains__('GNA'):
                    self.opt.step(b_x)
                elif str(self.opt).__contains__('LBFGS'):
                    self.opt.step(closure=closure)
                else:
                    self.opt.step()

                all_loss[epoch+1] = loss
                print('batch: {}, loss: {}'.format(batch_idx, loss.detach().numpy().item()))

        try:
            torch.save(self.net.state_dict(), Path.cwd() / 'result' / 'raw_train_fit_model.pth')
        except FileNotFoundError:
            pass
        
        self.net.eval()
        predict = self.net(self.X_test)
        predict = predict.data.numpy()
        
        if self.plt_opt:
            import matplotlib.pyplot as plt
            plt.scatter(self.X_test.numpy(), self.y_test, label='origin')
            plt.scatter(self.X_test.numpy(), predict, color='red', label='predict')
            plt.legend()
            plt.show()
    
    def gna_optimizer(self):

        for hessian_method in range(0, 1):
            self.opt = GNA(self.net.parameters(), lr=self.LR, model=self.net, hessian_approx=bool(hessian_method))
            self.test_optimizer()

    def adam_optimizer(self):
        
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.test_optimizer()

    def sgd_optimizer(self):

        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.LR)
        self.test_optimizer()

    def lbfgs_optimizer(self):

        self.opt = torch.optim.LBFGS(self.net.parameters(), lr=self.LR)
        self.test_optimizer()


    def test_all(self):
        print('\nGNA optimizer test')
        self.gna_optimizer()
        print('\nSGD optimizer test (for comparison)')
        self.sgd_optimizer()
        print('\nADAM optimizer test (for comparison)')
        self.adam_optimizer()
        print('\nLBFGS optimizer test (for comparison)')
        self.lbfgs_optimizer()


if __name__ == '__main__':
    unittest.main()