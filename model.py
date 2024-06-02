import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """ network constructor of the model"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.model_folder_path = './model'
        self.file_name_to_load = "model_best_120.pth"

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """ conections between layers """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """ save model to file"""
        self.model_folder_path = './model'
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        file_name = os.path.join(self.model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    def load(self):
        """ load previously saved model"""
        self.load_state_dict(torch.load(os.path.join(self.model_folder_path, self.file_name_to_load)))

class QTrainer:
    """
    class to make the model learn
    """
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.asarray(state), dtype=torch.float)
        next_state = torch.tensor(np.asarray(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:

            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()



