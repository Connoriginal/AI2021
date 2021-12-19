import argparse

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from model import NCF


from model import ModelClass
from utils import RecommendationDataset

if __name__ == '__main__':
    print("star")
    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--save-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./data', help='dataset directory')
    parser.add_argument('--batch-size', default=16, help='train loader batch size')
    parser.add_argument('--epoch-size', default=30, help='train epoch size')
    parser.add_argument('--lr', default=0.01, help='train learning rate')

    args = parser.parse_args()

    # load dataset in train folder
    train_data = RecommendationDataset(f"{args.dataset}/train.csv", train=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    valid_data = RecommendationDataset(f"{args.dataset}/valid.csv", train=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
    
    # Get details from datasets
    n_users, n_items, n_ratings = train_data.get_datasize()
    n_users_val, n_items_val, n_ratings_val = valid_data.get_datasize()

    # model = ModelClass(rank = 8)
    # model = BiasedMatrixFactorization(rank = 5)
    # model = GMF()
    model = NCF(num_users=610, num_items=193609, rank=5, num_layers=3, dropout=0.0, model="NCF")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device) # cuda

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay = 1e-4)
    criterion = nn.MSELoss()

    # array for train cost & valid cost
    train_cost = []
    valid_cost = []

    for epoch in range(args.epoch_size):
        cost = 0
        for users, items, ratings in train_loader:
            # cuda
            # users = users.cuda()
            # items = items.cuda()
            # ratings = ratings.cuda()

            optimizer.zero_grad()
            ratings_pred = model(users,items)
            loss = criterion(ratings_pred, ratings)
            loss.backward()
            optimizer.step()
            cost += loss.item() * len(ratings) # len은 loss.item()이 batch size의 평균이라 곱해준것
        
        cost /= n_ratings
        
        print(f"Epoch : {epoch}")
        print("train cost : {:.6f}".format(cost))
        
        cost_test = 0
        for users, items, ratings in valid_loader:
            # cuda
            # users = users.cuda()
            # items = items.cuda()
            # ratings = ratings.cuda()

            ratings_pred = model(users, items)
            loss = criterion(ratings_pred, ratings)
            cost_test += loss.item()*len(ratings)
        
        cost_test /= n_ratings_val 
        print("test cost : {:.6f}".format(cost_test))
        train_cost.append(cost)
        valid_cost.append(cost_test)

    # Figure

    # fig = plt.figure(figsize=(10,8))
    # # plt.title(f"k : {5}, learning rate : {0.001}, batch : {32}, weight decay {1e-5}")
    # plt.title(f"NCF model, k : {5}, layer : {3}, lr : {0.005}, weight decay : {1e-4}, drop out : {0.0}")
    # plt.plot(range(1,len(train_cost)+1),train_cost,label='Train Loss')
    # plt.plot(range(1,len(valid_cost)+1),valid_cost,label='Validation Loss')
    # plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    # plt.grid(True)
    # # plt.axvline(10, linestyle='-', color='r',label='Early Stopping Checkpoint')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.savefig("./img/FinalCheck3.png")
    # plt.show()


    torch.save(model.state_dict(), args.save_model)
