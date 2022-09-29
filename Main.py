from pickle import TRUE
import re
from traceback import print_tb
import torch
import torchvision.datasets as datasets
from torch import nn, optim
from tqdm import tqdm
from vae import VAE
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms


#configuration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20
NUM_EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 


#############################
########### DATASET #########
#############################


dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(),download=False)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
vae = VAE(INPUT_DIM,H_DIM, Z_DIM).to(DEVICE)
optimizer = optim.Adam(vae.parameters(),lr=LEARNING_RATE)
loss_fn = nn.BCELoss(reduction="sum")

### training_loop

for epoch in range(NUM_EPOCHS):
    l =  tqdm(enumerate(train_loader))
    for i, (x,_) in l:
        
        # forward pass
        x = x.to(DEVICE).view(x.shape[0],INPUT_DIM)
        x_decoder, mu, sigma = vae(x)

        # computing loss.....
        reconstruct_loss = loss_fn(x_decoder, x)
        kl = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # Backprop

        loss = reconstruct_loss + kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l.set_postfix(loss=loss.item())



vae = vae.to("cpu")
        
def inference(digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = vae.encoder(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = vae.decoder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=5)





