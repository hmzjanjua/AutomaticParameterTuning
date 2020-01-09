import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt


class NeuralRegressor(nn.Module):

    def __init__(self, n=3):
        super(NeuralRegressor, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.W1 = torch.ones(n, 4, dtype=torch.float).uniform_(50, 100).clone().detach().requires_grad_(True)
        self.b1 = torch.ones(1, 4, dtype=torch.float).uniform_(5, 10).clone().detach().requires_grad_(True)
        self.W2 = torch.ones(4, 2, dtype=torch.float).uniform_(50, 100).clone().detach().requires_grad_(True)
        self.b2 = torch.ones(1, 2, dtype=torch.float).uniform_(5, 10).clone().detach().requires_grad_(True)
        self.W3 = torch.ones(2, 1, dtype=torch.float).uniform_(50, 100).clone().detach().requires_grad_(True)
        self.b3 = torch.ones(1, 1, dtype=torch.float).uniform_(5, 20).clone().detach().requires_grad_(True)

        self.pars = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, d, h, c):
        df = pd.DataFrame()
        df['d'] = d
        df['h'] = h
        df['c'] = c

        x = torch.tensor(df.values, dtype=torch.float, device=self.device)
        x = torch.sigmoid(x.mm(self.W1).add(self.b1))
        x = torch.sigmoid(x.mm(self.W2).add(self.b2))
        x = torch.sigmoid(x.mm(self.W3).add(self.b3))
        return x


def forward_and_plot(inps):
    with torch.no_grad():
        d, h, c, s_hat = inps
        net = NeuralRegressor()
        s = net(d, h, c)

        # Plot predicted and actual
        plt.plot(range(len(s.data.tolist())), s.data.tolist(), 'b', label='predicted')
        plt.plot(range(len(s_hat.data.tolist())), s_hat.data.tolist(), 'r', label='actual')
        plt.legend(loc='upper left')
        plt.ylabel('SaleCount', color='black')
        plt.xlabel('Month', color='black')
        plt.title('SaleCount-Month')
        plt.gcf().canvas.set_window_title('SaleCount-Month')
        plt.gcf().set_size_inches(9, 5)
        plt.show()


def main():
    filename = '1_BAGS_BASLT.txt'
    filepath = 'E:/Dooney/Code/input/Dept_Class/Sales/Monthly/'
    filecompletepath = os.path.join(filepath, filename)
    epochs = 10000

    df = pd.read_csv(filecompletepath, sep='\t')
    df['month'] = [x[5:7] for x in df['FiscalMonthNumber']]
    df['year'] = [x[0:4] for x in df['FiscalMonthNumber']]
    df_train = df[(df['year'] == '2016') | (df['year'] == '2017')]
    df_test = df[(df['year'] == '2018') | (df['year'] == '2019')]
    print(f'Length of training data: {len(df_train)}')
    print(f'Length of testing data: {len(df_test)}')

    net = NeuralRegressor()

    d = torch.tensor(df_train['DiscountPerc'].values, dtype=torch.float)
    h = torch.tensor(df_train['Holiday'].values, dtype=torch.float)
    c = torch.tensor(df_train['Campaign'].values, dtype=torch.float)
    s_hat = torch.tensor(df_train['SaleCount'].values, dtype=torch.float)

    optimizer = optim.Adam(net.pars, lr=0.1)

    loss_arr = []
    lowest_loss = torch.tensor(float('inf'), dtype=torch.float)
    try:
        for i in range(epochs):
            if i % 1000 == 0:
                print(f"Done: {i/epochs * 100} %")
                print("Current Lowest Loss: " + '{:.2e}'.format(lowest_loss.data.tolist()))
            s = net(d, h, c)

            criterion = nn.MSELoss()
            loss = criterion(s, s_hat)
            loss_arr.append(loss.data.tolist())

            if loss < lowest_loss:
                lowest_loss = loss

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        pass

    print(f"lowest loss: {lowest_loss}")

    plt.plot(range(1, epochs+1), loss_arr)
    plt.title('Loss-Epoch')
    plt.ylabel('loss', color='black')
    plt.xlabel('epochs', color='black')
    plt.gcf().canvas.set_window_title('Loss-Epoch')
    plt.gcf().set_size_inches(9, 5)
    plt.show()

    d = torch.tensor(df_test['DiscountPerc'].values, dtype=torch.float)
    h = torch.tensor(df_test['Holiday'].values, dtype=torch.float)
    c = torch.tensor(df_test['Campaign'].values, dtype=torch.float)
    s_hat = torch.tensor(df_test['SaleCount'].values, dtype=torch.float)

    forward_and_plot((d, h, c, s_hat))


if __name__ == '__main__':
    main()
