import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *
from torchvision.utils import save_image

codings_size = 100
EPSILON = 0.00005
dropout_p = 0.4
LEARNING_RATE = 0.0005
BATCH_SIZE = 512 #160 #128
N_EPOCHS = 20 # 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ae2_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def plot_one_image(image):
    image = np.squeeze(image, axis=0)
    plt.imshow(image.permute(1, 2, 0))  # Assuming image is of shape (C, H, W)
    plt.axis("off")

def plot_image_pair(original_image, reconstructed_image):
    original_image = np.squeeze(original_image, axis=0)
    reconstructed_image = np.squeeze(reconstructed_image, axis=0)
    plt.subplot(1, 2, 1)
    plt.imshow(original_image.permute(1, 2, 0))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image.permute(1, 2, 0))
    plt.axis("off")


def plot_multiple_images(images_batch, n, n_init, model):
    n_cols = n
    n_rows = 2
    plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    for i in range(n):
        index = i + n_init
        image = images_batch[index].unsqueeze(0)
        rec_image = model(image).cpu().detach().numpy()  # Move tensor to CPU before calling .detach().numpy()
        image = np.squeeze(image.cpu(), axis=0).permute(1, 2, 0)   # Move tensor to CPU and change shape (C, H, W) to (H, W, C)
        rec_image = np.squeeze(rec_image, axis=0).transpose((1, 2, 0)) # Change shape (C, H, W) to (H, W, C)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(n_rows, n_cols, i + 1 + n)
        plt.imshow(rec_image)
        plt.axis("off")


class Encoder(nn.Module):
    def __init__(self, codings_size=100, dropout_p=0.4):
        super(Encoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, eps=EPSILON),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128, eps=EPSILON),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256, eps=EPSILON),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),

            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(512, eps=EPSILON),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),

            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024, eps=EPSILON),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p)
        )

        self.fc = nn.Sequential(
            nn.Linear(1024 * 8 * 8, codings_size),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, codings_size):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(codings_size, 8 * 8 * 1024)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1024, 8, 8)  # Reshape to 4D tensor
        x = self.relu(x)

        x = self.conv_layers(x)

        return x

def train_autoencoder2(data_path, checkpoint_dir, read_from_disk, resume_dir, get_from_checkpoint):
    # Create an instance of the encoder
    encoder = Encoder()
    # Instantiate the decoder
    decoder = Decoder(codings_size)

    # Combine encoder and decoder to autoencoder
    autoencoder = nn.Sequential(encoder, decoder)
    autoencoder.to(device)
    # Define optimizer
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

    checkpoint_name = 'ae2.pth'
    if read_from_disk:
        print('reading ae from disk')
        if resume_dir is not None:
            checkpoint_filename = os.path.join(resume_dir, checkpoint_name)
            if os.path.exists(checkpoint_filename):
                autoencoder, optimizer, best_epoch, best_acc = (
                    load_model_from_checkpoint(checkpoint_filename, autoencoder, optimizer))
                autoencoder.to(device)
                if get_from_checkpoint:
                    print(f'train_aotoencoder2 returning model from {checkpoint_filename}')
                    return autoencoder

    # Define transforms

    # Create dataset and data loader
    #data_path = r'D:\datasets\celebs_100k'
    dataset = datasets.ImageFolder(root=data_path, transform=ae2_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize variables to track best accuracy and corresponding epoch
    t = time.localtime()
    # checkpoint_dir = r'.\checkpoints\autoencoder2\incremental'
    sample_images_dir = os.path.join(checkpoint_dir, 'sample_images')

    # Check if a checkpoint exists and load it
    # Check if a checkpoint exists and load it
    since = time.time()
    best_loss = float('inf')

    # Define loss function
    criterion = nn.MSELoss()
    criterion.to(device)


    # Training loop
    for epoch in range(N_EPOCHS):
        print(f'training epoch {epoch}')
        epoch_start_time = time.time()

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f'count {i} loss {loss.item()}')
            # if i % 100==99:  # Print every 100 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 100))
            #     running_loss = 0.0
            #     # display some images
            #     plot_multiple_images(inputs,n=8, n_init=200, model=autoencoder)

        # Print average training loss for the epoch
        epoch_train_loss = running_loss / len(data_loader.dataset)
        print(f"Epoch [{epoch + 1}/{N_EPOCHS}], Train Loss: {epoch_train_loss:.4f}")
        epoch_elapsed_time = time.time() - epoch_start_time
        print(f'Epoch time {epoch_elapsed_time // 60:.0f}m {epoch_elapsed_time % 60:.0f}s')

        # get images after each epoch
        generated_images = autoencoder(inputs)
        generated_images = generated_images[:20,:,:,:]
        generated_images = generated_images.view(-1, 3, 128, 128)

        # Clip values to ensure they're in [0, 1] range
        generated_images = torch.clamp(generated_images, 0, 1)

        # Save the batch of images using torchvision.utils.save_image
        save_image(generated_images,
                   os.path.join(sample_images_dir, f'generated_images_epoch_{epoch + 1}.png'), nrow=5)


        val_images = torch.clamp(inputs, 0, 1)
        val_images = val_images[:20,:,:,:]

        save_image(val_images, os.path.join(sample_images_dir, f'original_images.png'), nrow=5)

        if epoch_train_loss < best_loss:
            best_loss = epoch_train_loss
            best_epoch = epoch

            # Save checkpoint
            checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save({
                'model_state_dict': autoencoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_loss,
            }, checkpoint_filename)
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            print(current_time)
            print(f"saved checkpoint best epoch: {best_epoch}, best_acc: {best_loss}")

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best Loss: {best_loss:.4f}')


    print('Finished Training')
    return autoencoder

