import argparse

import torch
from torch.utils.data import DataLoader

from model import NCF
from utils import RecommendationDataset


def inference(data_loader, model):
    """ model inference """

    model.eval()
    preds = []

    with torch.no_grad():
        for users, items in data_loader:
            predicted = model(users, items)

            preds.extend(map(lambda t: t.item(), predicted))

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='2021 AI Final Project')
    parser.add_argument('--load-model', default='model.pt', help="Model's state_dict")
    parser.add_argument('--dataset', default='./test', help='dataset directory')
    parser.add_argument('--batch-size', default=16, help='test loader batch size')

    args = parser.parse_args()

    # run device
    device = torch.device('cpu')
    # instantiate model
    model = NCF(num_users=610, num_items=193609, rank=5, num_layers=3, dropout=0.0, model="NCF")
    model.load_state_dict(torch.load(args.load_model,map_location=device))

    # load dataset in test folder
    test_data = RecommendationDataset(f'{args.dataset}/ratings.csv', train=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # write model inference
    preds = inference(test_loader, model)

    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))
    
