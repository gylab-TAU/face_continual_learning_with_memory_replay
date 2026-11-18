import os
import shutil

import numpy as np
import pickle

from arcface_resnet_train import *
from utils import *
from Memory import Memory
from torchvision.utils import save_image
from autoencoder_train2 import *
from scipy.stats import chi2

AE_MEMORY_CLUSTERING_RADIUS = 0.2
EMBD_MEMORY_CLUSTERING_RADIUS = 0.1
AE_MEMORY_CLUSTERING_DISTANCE_TH = 0.2
EMBD_MEMORY_CLUSTERING_DISTANCE_TH = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Baby:
    def __init__(self):
        self.embd_model = None
        self.autoencoder = None
        # Initialize an empty memory dictionary
        self.embd_memory = Memory()
        self.ae_memory = Memory()

    def save(self, baby_path):
        os.makedirs(baby_path, exist_ok=True)

        # Save memories
        self.embd_memory.save_memory(os.path.join(baby_path, 'embd_memory'))
        self.ae_memory.save_memory(os.path.join(baby_path, 'ae_memory'))

    def load_from_disk(self, baby_path):
        # Load memories
        self.embd_memory.load_memory(os.path.join(baby_path, 'embd_memory'))
        self.ae_memory.load_memory(os.path.join(baby_path, 'ae_memory'))

    def train(self, dataloaders, train_method, model_type, num_epochs, embd_len,
              pretrained, prev_classes, current_classes, get_from_checkpoint,
              checkpoint_name='checkpoint.pth',opt = None):

        if train_method == 'arcface_resnet':
            self.embd_model = train_arcface_resnet(opt, dataloaders, get_from_checkpoint)

        else:
            raise ValueError(f'train method {train_method} not supported')

    def train_ae2(self,data_path,checkpoints_path, read_from_disk = False, resume_dir = '', get_from_checkpoint = False):
        if get_from_checkpoint:
            self.autoencoder = train_autoencoder2(data_path,checkpoints_path, read_from_disk, resume_dir, get_from_checkpoint)
        else:
            train_autoencoder2(data_path, checkpoints_path, read_from_disk, resume_dir, get_from_checkpoint)
            # after training read the best autoencoder
            checkpoint_dir = checkpoints_path
            checkpoint_name = 'ae2.pth'
            checkpoint_filename = os.path.join(checkpoint_dir, checkpoint_name)
            if os.path.exists(checkpoint_filename):
                encoder = Encoder()
                # Instantiate the decoder
                decoder = Decoder(codings_size)
                # Combine encoder and decoder to autoencoder
                autoencoder = nn.Sequential(encoder, decoder)
                # Define optimizer
                optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

                autoencoder, optimizer, best_epoch, best_acc = (
                    load_model_from_checkpoint(checkpoint_filename, autoencoder, optimizer))

                self.autoencoder = autoencoder

    def generate_images_from_ae2(self, data_path, class_names, clear_prev = True):

        samples = self.ae_memory.reconstruct_from_centroids_and_stds()
        for key, val in samples.items():
            if key in class_names:
                class_dir = os.path.join(data_path,key)
                print(f'generate_images_from_ae2 creating for class {key}')
                if clear_prev:
                    if os.path.isdir(class_dir):
                        shutil.rmtree(class_dir)
                os.makedirs(class_dir, exist_ok=True)

                i = 0
                j = 0
                while i < len(val):
                    ss = val[i:i + 20].astype(np.float32)
                    ss = torch.from_numpy(ss).to(device)
                    ims = self.autoencoder[1](ss)
                    generated_images = ims.view(-1, 3, 128, 128)

                    # Clip values to ensure they're in [0, 1] range
                    generated_images = torch.clamp(generated_images, 0, 1)
                    for im in generated_images:
                        save_image(im,os.path.join(class_dir, f'g_{j}.png'))
                        j = j+1
                    i += 20

    def build_embd_memory(self, dataloader, clear_prev_memory = False, clear_prev_ids = False, is_familiar = False):
        if self.embd_model is None:
            print('baby embd_model is none, failing build_embd_memory')
            return

        self.embd_model.eval()
        self.embd_model.to(device)
        if clear_prev_memory:
            self.embd_memory = None

        if self.embd_memory is None:
            self.embd_memory = Memory()

        if hasattr(dataloader.dataset, 'dataset'):
            class_to_idx = dataloader.dataset.dataset.dataset.class_to_idx
        else:
            class_to_idx = dataloader.dataset.class_to_idx

        if clear_prev_ids:
            # clear the memory of the labels in the dataset
            for class_name in class_to_idx.keys():
                self.embd_memory.clear_record(class_name)

        # optional output
        class_embeddings = {}
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                output = self.embd_model(inputs)
                embeddings = output
                for embedding, label in zip(embeddings.cpu().numpy(), labels.numpy()):
                    class_name = [i for i in class_to_idx if class_to_idx[i] == label]
                    self.embd_memory.add_embedding_to_class(embedding, class_name[0],is_familiar=is_familiar)
                    if class_name[0] not in class_embeddings.keys():
                        class_embeddings[class_name[0]] = []
                    class_embeddings[class_name[0]].append(embedding)

        self.embd_memory.calc_mean_std(clustering_radius=EMBD_MEMORY_CLUSTERING_RADIUS, 
                                       clustering_distance_threshold=EMBD_MEMORY_CLUSTERING_DISTANCE_TH)
        return class_embeddings

    def build_ae_memory(self,data_path, clear_prev_memory = False, clear_prev_ids = False):

        dataset = datasets.ImageFolder(root=data_path, transform=ae2_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        if self.autoencoder is None:
            print('baby autoencoder is none, failing build_ae_memory')
            return

        self.autoencoder.eval()
        self.autoencoder.to(device)
        if clear_prev_memory:
            self.ae_memory = None

        if self.ae_memory is None:
            self.ae_memory = Memory()

        class_to_idx = dataset.class_to_idx
        if clear_prev_ids:
            # clear the memory of the labels in the dataset
            for class_name, class_label in class_to_idx.items():
                self.ae_memory.clear_record(class_name)

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                embds = self.autoencoder[0](inputs)
                embds = embds.detach()
                for embedding, label in zip(embds.cpu().numpy(), labels.numpy()):
                    class_name = [i for i in class_to_idx if class_to_idx[i] == label]
                    self.ae_memory.add_embedding_to_class(embedding, class_name[0])

        self.ae_memory.calc_mean_std(clustering_radius=AE_MEMORY_CLUSTERING_RADIUS, 
                                     clustering_distance_threshold=AE_MEMORY_CLUSTERING_DISTANCE_TH)

    def test(self, dataloader, data_subset, class_names,
             performance_df):
        per_class_accuracy, overall_accuracy, true_labels, predicted_labels = \
            evaluate_accuracy(self.embd_model, dataloader, self.embd_memory)

        performance_df.loc[len(performance_df.index)] = ['overall', overall_accuracy]
        print(f'overall accuracy: {overall_accuracy}')
        for key, val in per_class_accuracy.items():
            print(f'class {key} accuracy {val}')
            performance_df.loc[len(performance_df.index)] = [key, val]
        # show_examples(data_subset, predicted_labels, class_names)
        return performance_df

    def test_verification(self, pair_file):

        evaluate_verification_accuracy(self.embd_model,pairs_file=pair_file)

    def evaluate_mse(self, data_path):

        dataset = datasets.ImageFolder(root=data_path, transform=ae2_transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        if self.autoencoder is None:
            print('baby autoencoder is none, failing evaluate_mse')
            return

        self.autoencoder.eval()
        self.autoencoder.to(device)

        class_to_idx = dataset.class_to_idx

        # Create dict: { class_name : [mse1, mse2, ...] }
        mse_per_class = {cls_name: [] for cls_name in class_to_idx.keys()}

        mse_loss = nn.MSELoss(reduction='none')

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                recon = self.autoencoder(inputs)

                # Pixelwise squared error
                mse = mse_loss(recon, inputs)

                # Per image MSE
                mse_per_image = mse.view(mse.size(0), -1).mean(dim=1).cpu().numpy()

                # Assign each image's mse to the correct class
                for mse_value, label in zip(mse_per_image, labels.numpy()):
                    class_name = [name for name, idx in class_to_idx.items() if idx == label][0]
                    mse_per_class[class_name].append(float(mse_value))

        return mse_per_class

