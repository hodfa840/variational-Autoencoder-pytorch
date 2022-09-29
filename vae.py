import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, h_dim=200, z_dim=64) :
        super().__init__()
        #encoder 
        self.img_to_hidden = nn.Linear(input_dim, h_dim)
        self.hidden_to_mu = nn.Linear(h_dim, z_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, z_dim)

        # decoder
        self.z_to_hidden = nn.Linear(z_dim, h_dim)
        self.hidden_to_img = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU()


    def encoder(self, x):

        h = self.relu(self.img_to_hidden(x))
        mu , sigma = self.hidden_to_mu(h), self.hidden_to_sigma(h) 
        return mu, sigma

    
    def decoder(self,z):

        h = self.relu(self.z_to_hidden(z))
        I = torch.sigmoid(self.hidden_to_img((h)))

        return I 


    def forward(self,x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(mu)
        z_new = mu + sigma * epsilon
        x_recunstructed = self.decoder(z_new)
        return x_recunstructed, mu, sigma



if __name__ == '__main__':

    x = torch.rand(5, 28* 28)
    vae = VAE(input_dim=28*28)
    x__decoder , mu , sigma = vae(x)

    print(x__decoder.shape)
    print(mu.shape)
    print(sigma.shape)