import torch.nn as nn
import torch


class MetaOptimizer(nn.Module):
    # this model just uses a simple feed-forward network to obtain the next set of parameters
    # the problem is that activation function used in the intermediate layers results in only
    # positive values between 0 and 1
    # the slope can never be negative for the separating hyperplane
    # alternative would be to use tanh as the activation function which generates values between -1 and 1
    # However the problem is in constructing the loss function which is log of the policy function and hence
    # outputting negative value leads to log of a negative number which produces NaN
    def __init__(self, input_size, hidden_size):
        super(MetaOptimizer, self).__init__()
        self.input_size = input_size
        self.l2h = nn.Linear(input_size, hidden_size)
        self.actv = nn.Tanh()
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2i = nn.Linear(hidden_size, input_size)

    def forward(self, inp):
        x = self.l2h(inp)
        x = self.actv(x)
        x = self.h2h(x)
        x = self.actv(x)
        x = self.h2i(x)
        return self.actv(x)


class MetaOptimizerDirection(nn.Module):
    # instead of outputting the next set of parameters this model tries to predict the direction of descent
    # the problem is mapped to a 2-class classification problem
    def __init__(self, output_size=2, hidden_size=10, layers=1):
        super(MetaOptimizerDirection, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.l2h = nn.Linear(1, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, layers)
        self.h2o = nn.Linear(hidden_size, output_size) # 2-class problem either increase or decrease
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp):
        c0, h0 = self.init_hidden()
        x = self.l2h(inp.reshape(-1, 1))
        x = self.relu(x)
        x = x.reshape(-1, 1, self.hidden_size)
        out, _ = self.rnn(x, (h0, c0))
        out = out.squeeze(1)
        out = self.h2o(out)
        out = self.sigmoid(out)
        out = self.softmax(out)
        return out

    def init_hidden(self):
        c0 = torch.randn(self.layers, 1, self.hidden_size)
        h0 = torch.randn(self.layers, 1, self.hidden_size)
        return c0, h0


class MetaOptimizerMLP(nn.Module):
    def __init__(self, hidden_size=20, output_size=2):
        super(MetaOptimizerMLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.LogSigmoid(),
            nn.Softmax(dim=1)
        )

    def forward(self, inp):
        x = torch.t(inp)
        out = self.layer(x)
        return out


if __name__ == '__main__':
    inp = torch.FloatTensor([[-0.0222, -0.7004,  0.7234, -0.1074,  0.3646,  0.7918, -0.1753, -0.3265]])
    model = MetaOptimizerDirection(hidden_size=10, layers=2)
    #model = MetaOptimizerMLP(hidden_size=20, output_size=2)
    out = model(inp)
    print(out)
    print(torch.argmax(out, dim=1))