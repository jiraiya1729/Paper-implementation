import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transform as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.lin1 = nn.Linear(z_dim, 7*7*64)
        self.ct1 = nn.ConvTranspose2d(64, 32, stride=2)
        self.ct2 = nn.ConvTranspose2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)
        x = self.ct1(x)
        x = F.relu(x)
        x = self.ct2(x)
        x = F.relu(x)
        return x
   
    
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50


disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.normalize((0.1307,), (0.3081,))]    
)

dataset = datasets.MNIST(root = "dataset/", transform = transforms, download = True)
loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()


writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator: to maximize log(D(real)) + log(1 - D(G(z)))

        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        
        disc_real = disc(real).view(-1) 
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) 

        disc_fake = disc(fake.detach()).view(-1) # this makes sure the image is detached from the computation graph or computing gradients
        lossD_fake = criterion(disc_fake, torch.ones_like(disc_fake))

        lossD = (lossD_fake + lossD_real) / 2
        
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()
        
        
        # Train Generator: to minimize log(1 - D(G(z))) <==> maximize log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Loss D: {lossD:..4f}, Loss G: {lossG:.4f}"
            )
            
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize = True)
                img_grid_real = torchvision.utils.make_grid(real, normalize = True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_fake, global_step = step
                )
                
                writer_real.add_image(
                    "MNIST real Images", img_grid_real, global_step = step
                )
                
                step += 1
        



        