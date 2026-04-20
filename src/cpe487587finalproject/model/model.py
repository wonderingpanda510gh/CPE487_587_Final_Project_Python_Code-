import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# first, we define the variational autoencoder model, which consists of an encoder and a decoder. And also we need to define the reparameterization trick, which allows us to do the backpropagation
class VAE(nn.Module):
    def __init__(self, cat_dims, num_continuous, latent_dim=16):
        super(VAE, self).__init__()
        
        # entity embedding for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=max(2, dim // 2)) 
            for dim in cat_dims
        ])
        
        # compute the total input dimension for the encoder, wihich consists of the embedded categorical features and the continuous features
        total_emb_dim = sum([max(2, dim // 2) for dim in cat_dims])
        self.input_dim = total_emb_dim + num_continuous
        
        # encoder
        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # decoder
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, self.input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4)

    def forward(self, x_cat, x_num):
        # embedding categorical features
        embs = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)
        
        # concatenate embedded categorical features with continuous features
        x = torch.cat([x_emb, x_num], dim=1)
        
        # encode, reparameterize, and decode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_x = self.decode(z)
        
        return decoded_x, x, mu, logvar, z

# the total loss function for our vae, we define the mse loss and the kl divergence loss, we use mse because our data is not binary
def vae_loss_function(decoded_x, x, mu, logvar):
    mse = F.mse_loss(decoded_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kl


# here we define the feature tokenize transformer (ft-transformer)
class FeatureTokenizer(nn.Module):
    def __init__(self, cat_dims, num_continuous, token_dim):
        super(FeatureTokenizer, self).__init__()
        # we first map the continuous features to token_dim
        self.weight = nn.Parameter(torch.Tensor(num_continuous, token_dim))
        self.bias = nn.Parameter(torch.Tensor(num_continuous, token_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # using kaiming initialization for the weight
        nn.init.zeros_(self.bias) # bias is initialized to zero
        
        # then we create an embedding layer for each categorical feature, which maps the categorical features to token_dim
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(dim, token_dim) for dim in cat_dims
        ])

    def forward(self, x_cat, x_num):
        # x_num shape: (batch_size, num_continuous) to (batch_size, num_continuous, token_dim)
        x_num_tokens = self.weight.unsqueeze(0) * x_num.unsqueeze(-1) + self.bias.unsqueeze(0)
        
        # x_cat tokens
        cat_tokens = [emb(x_cat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        x_cat_tokens = torch.cat(cat_tokens, dim=1)
        
        # return the concatenated tokens for both categorical and continuous features, the shape will be (batch_size, num_features, token_dim)
        return torch.cat([x_cat_tokens, x_num_tokens], dim=1)

class FTTransformer(nn.Module):
    def __init__(self, cat_dims, num_continuous, token_dim=32, num_heads=4, num_layers=3, num_targets=2):
        super(FTTransformer, self).__init__()
        
        self.tokenizer = FeatureTokenizer(cat_dims, num_continuous, token_dim)
        
        # define the classification token, which will be used to aggregate the information from all the tokens for prediction
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        
        # transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # layerNorm and prediction layers for prediction
        self.layer_norm = nn.LayerNorm(token_dim)
        self.predict = nn.Sequential(
            nn.Linear(token_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_targets) # the output for ExamScore and FinalGrade
        )

    def forward(self, x_cat, x_num):
        batch_size = x_num.size(0)
        tokens = self.tokenizer(x_cat, x_num)
        
        # combine the classification token with the feature tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        
        # use transformer to process the tokens
        x = self.transformer(x)
        
        # get the output from the classification token, which will be used for prediction
        cls_out = x[:, 0, :]
        cls_out = self.layer_norm(cls_out)
        
        return self.predict(cls_out)