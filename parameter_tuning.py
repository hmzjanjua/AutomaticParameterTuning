import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt


class AutomaticParameterTuning(nn.Module):

    def __init__(self, m_d=1000.0, c_d=1000.0, k=2.5, a=0.25, m_h=1000.0, c_h=1000.0):
        super(AutomaticParameterTuning, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = 'cpu'

        self.m_d = torch.tensor(m_d, requires_grad=True, dtype=torch.float)
        self.c_d = torch.tensor(c_d, requires_grad=True, dtype=torch.float)
        self.k = torch.tensor(k, requires_grad=True, dtype=torch.float)
        self.a = torch.tensor(a, requires_grad=True, dtype=torch.float)
        self.m_h = torch.tensor(m_h, requires_grad=True, dtype=torch.float)
        self.c_h = torch.tensor(c_h, requires_grad=True, dtype=torch.float)

        self.pars = [self.m_d, self.c_d, self.k, self.a, self.m_h, self.c_h]

    def forward(self, d, h, c):
        s_d = self.m_d.mul(d).add(self.c_d)
        s_c = (1.0 - torch.exp(-self.a.mul(c))) * self.k
        s_h = self.m_h.mul(h).add(self.c_h)

        s = s_d*s_c + s_d + s_h

        return s


def forward_and_plot(inps, lowest_pars):
    with torch.no_grad():
        d, h, c, s_hat = inps
        apt = AutomaticParameterTuning(
            lowest_pars[0], lowest_pars[1], lowest_pars[2], lowest_pars[3], lowest_pars[4], lowest_pars[5]
        )
        s = apt(d, h, c)

        # Plot predicted and actual
        plt.plot(range(len(s.data.tolist())), s.data.tolist(), 'b', label='predicted')
        plt.plot(range(len(s_hat.data.tolist())), s_hat.data.tolist(), 'r', label='actual')
        plt.legend(loc='upper left')
        plt.ylabel('Sale', color='black')
        plt.xlabel('Month', color='black')
        plt.title('Sale-Month')
        plt.gcf().canvas.set_window_title('Sale-Month')
        plt.gcf().set_size_inches(9, 5)
        plt.show()


def main():
    filenames = ['1_BAGS_BASLT.txt', '2_SLG_BASLT.txt', '3_BAGS_NYLON.txt', '4_BAGS_EXOTC.txt', '5_BAGS_NOVEL.txt',
                 '6_BAGS_FASHL.txt', '7_BAGS_ALTO.txt', '8_BAGS_FASHF.txt', '9_BAGS_NFL.txt', '10_BAGS_MLB.txt']
    filenumber = 3
    filename = filenames[filenumber-1]
    filepath = 'E:/Dooney/Code/AutomaticParameterTuning/level4/dus/'
    filecompletepath = os.path.join(filepath, filename)
    configfilename = 'config.csv'
    epochs = 100000

    configfile = pd.read_csv(os.path.join(filepath, configfilename), sep=',')
    config = configfile[configfile['filename'] == filename]
    assert (len(config) == 1)

    m_d = float(config['gradient'].astype(float))
    c_d = float(config['intercept'].astype(float))
    m_h = float(config['holidayfactor'].astype(float))
    c_h = float(config['holidayconstant'].astype(float))
    k = float(config['campaignfactor'].astype(float))
    a = float(config['campaignexponentfactor'].astype(float))

    df = pd.read_csv(filecompletepath, sep='\t')
    df['month'] = [x[5:7] for x in df['Month']]
    df['year'] = [x[0:4] for x in df['Month']]
    df_train = df[((df['year'] == '2018') | (df['year'] == '2019')) & (df['Sale'] > 0.0)]
    df_test = df[(df['year'] == '2018') | (df['year'] == '2019')]
    print(f'Length of training data: {len(df_train)}')
    print(f'Length of testing data: {len(df_test)}')

    apt = AutomaticParameterTuning(m_d=m_d, c_d=c_d, k=k, a=a, m_h=m_h, c_h=c_h)

    d = torch.tensor(df_train['DiscountPerc'].values, dtype=torch.float)
    h = torch.tensor(df_train['Holiday'].values, dtype=torch.float)
    c = torch.tensor(df_train['Campaign'].values, dtype=torch.float)
    s_hat = torch.tensor(df_train['Sale'].values, dtype=torch.float)

    optimizer = optim.Adam(apt.pars, lr=0.1)

    loss_arr = []
    lowest_loss = torch.tensor(float('inf'), dtype=torch.float)
    lowest_pars = apt.pars
    try:
        for i in range(epochs):
            if (i % 1000 == 0) and (i > 0):
                print(f"Done: {i/epochs * 100} %")
                print("Current Lowest Loss: " + '{:.2e}'.format(lowest_loss.data.tolist()))
                loss_arr.append(loss.data.tolist())

            s = apt(d, h, c)

            # e = (s - s_hat) / s_hat
            # e_hat = torch.zeros(e.numel(), dtype=torch.float)

            criterion = nn.MSELoss()
            loss = criterion(s, s_hat)
            # loss_arr.append(loss.data.tolist())

            if loss < lowest_loss:
                lowest_loss = loss
                lowest_pars = []
                for par in apt.pars:
                    lowest_pars.append(par.data.tolist())

            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        pass

    print(f"initial pars:\n{[m_d, c_d, k, a, m_h, 0]}")

    print(f"lowest loss: {lowest_loss}")
    print(f"lowest pars:\n{lowest_pars}")

    plt.plot(range(1, len(loss_arr)+1), loss_arr)
    plt.title('Loss-Epoch')
    plt.ylabel('loss', color='black')
    plt.xlabel('epochs', color='black')
    plt.gcf().canvas.set_window_title('Loss-Epoch')
    plt.gcf().set_size_inches(9, 5)
    plt.show()

    d = torch.tensor(df_test['DiscountPerc'].values, dtype=torch.float)
    h = torch.tensor(df_test['Holiday'].values, dtype=torch.float)
    c = torch.tensor(df_test['Campaign'].values, dtype=torch.float)
    s_hat = torch.tensor(df_test['Sale'].values, dtype=torch.float)

    forward_and_plot((d, h, c, s_hat), lowest_pars)


if __name__ == '__main__':
    main()

    # s = apt(d_i, h_i, c_i)  # (43.7314050642448, 3, 14)
    # s_hat = torch.tensor(s_hat_i, dtype=torch.float)  # (11759275.9300)

    # e = (s - s_hat) / s_hat
    # print(f"Error: {e * 100}%")

    # with torch.no_grad():
    #     s = apt(d, h, c)
    #     df['s'] = s
    #
    #     print(df[['s', 'Sale']])

    # lr = 0.000000000000001
    # for par in apt.pars:
    #     par.data.sub_(par.grad * lr)
