import torch
from torch import nn

# MF model with nn.Parameter
class ModelClass(nn.Module):
    def __init__(self, num_users=610, num_items=193609, rank=10):
        super().__init__()
        self.U = torch.nn.Parameter(torch.randn(num_users + 1, rank))
        self.V = torch.nn.Parameter(torch.randn(num_items + 1, rank))

    def forward(self, users, items):
        ratings = torch.sum(self.U[users] * self.V[items], dim=-1)
        return ratings

# MF model with nn.Embedding
class BiasedMatrixFactorization(nn.Module):
    def __init__(self, num_users=610, num_items=193609, rank=10):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users+1, rank)
        self.item_factors = torch.nn.Embedding(num_items+1, rank)
        self.user_biases = torch.nn.Embedding(num_users+1, 1)
        self.item_biases = torch.nn.Embedding(num_items+1, 1)

    def forward(self, user, item):
        pred = self.user_biases(user) + self.item_biases(item)
        pred += (
            (self.user_factors(user) * self.item_factors(item))
            .sum(dim=1, keepdim=True)
        )
        return pred.squeeze()

# Generalized MF model from https://arxiv.org/pdf/1708.05031.pdf
class GMF(nn.Module):
    def __init__(
        self, num_users=610, num_items=193609, rank=10,
    ):
        super(GMF, self).__init__()
        self.factor_num = rank

        # 임베딩 저장공간 확보; num_embeddings, embedding_dim
        self.embed_user = nn.Embedding(num_users+1, rank)
        self.embed_item = nn.Embedding(num_items+1, rank)
        predict_size = rank
        self.predict_layer = nn.Linear(predict_size,1)
        self._init_weight_()

    def _init_weight_(self):
        # weight 초기화
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")


        # bias 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        output_GMF = embed_user * embed_item
        concat = output_GMF

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

# NCF model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, rank, num_layers, dropout, model):
        super(NCF, self).__init__()
        """
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF';
		"""		
        self.dropout = dropout
        self.model = model

        self.embed_user_GMF = nn.Embedding(num_users+1, rank)
        self.embed_item_GMF = nn.Embedding(num_items+1, rank)
        self.embed_user_MLP = nn.Embedding(num_users+1, rank * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(num_items+1, rank * (2 ** (num_layers - 1)))

        MLP_modules = []
        for i in range(num_layers):
            input_size = rank * (2 ** (num_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = rank 
        else:
            predict_size = rank * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)