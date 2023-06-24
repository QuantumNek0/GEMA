from config import *
import pandas as pd
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 150
from sklearn.manifold import TSNE

from classes.midi import MIDI
from classes.autoencoder import *


def to_array(df: pd.DataFrame) -> np.array:

    arr = []
    for item in df:
        arr += [item]

    return np.array(arr)


# def latent_to_df(autoencoder: VariationalAutoencoder,
#                  data_loader: torch_data.DataLoader,
#                  num_batches: int = 100) -> pd.DataFrame:
#     latent_df = pd.DataFrame()
#
#     latent_points = []
#     latent_labels = []
#     for i, (file_batch, label_batch) in tqdm(enumerate(data_loader), desc="converting batches", unit="batch"):
#         for file, label in zip(file_batch, label_batch):
#
#             encoded_file = autoencoder.encoder.encoder(file)
#             encoded_file = encoded_file.flatten().detach().numpy()
#             latent_points += [encoded_file]
#             latent_labels += [label.numpy()]
#
#         if i > num_batches:
#             break
#
#     latent_df["points"] = latent_points
#     latent_df["labels"] = latent_labels
#
#     return latent_df

def latent_to_df(autoencoder: VariationalAutoencoder,
                 data_loader: torch_data.DataLoader,
                 num_batches: int = 100) -> pd.DataFrame:
    latent_df = pd.DataFrame()

    latent_points = []
    latent_labels = []
    for i, (file_batch, label_batch) in tqdm(enumerate(data_loader), desc="converting batches", unit="batch", total=num_batches):

        encoded_batch = autoencoder.encoder(file_batch)
        for encoded_file, label in zip(encoded_batch, label_batch):

            encoded_file = encoded_file.detach().numpy()
            latent_points += [encoded_file]
            latent_labels += [label.numpy()]

        if i > num_batches:
            break

    latent_df["points"] = latent_points
    latent_df["labels"] = latent_labels

    return latent_df


def main():
    from sklearn.decomposition import PCA

    latent_dims = DEFAULT_LATENT_SIZE

    midi_dataset = MidiDataset("classes/data/augmented_data")
    data_loader = torch_data.DataLoader(midi_dataset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)

    # data = torch.stack([midi_dataset[i][0] for i in range(len(midi_dataset))])
    # std_data = (data - data.mean(dim=0)) / data.std(dim=0)

    # pca = PCA(n_components=2)
    # pca_data = pca.fit_transform(std_data)

    # plt.scatter(pca_data[:, 0], pca_data[:, 1])
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.show()

    vae = VariationalAutoencoder(latent_dims)
    vae.load_state_dict(torch.load("classes/music_vae.pt"))

    latent_df = latent_to_df(vae, data_loader)

    if latent_dims > 2:
        print(f"applying TSNE transformation...")

        points = to_array(latent_df["points"])

        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(points)

        x = tsne_results[:, 0].tolist()
        y = tsne_results[:, 1].tolist()

        plt.scatter(x, y, c=latent_df["labels"])
        plt.colorbar()
        plt.show()

    else:

        x = []
        y = []

        for point in latent_df["points"]:
            x += [point[0]]
            y += [point[1]]

        plt.scatter(x, y, c=latent_df["labels"])
        plt.colorbar()
        plt.show()

    # mid = MIDI(single_notes=True)
    #
    # mid.read_midi("classes/data/original_data/Avicii_Levels-12B_126.mid")
    #
    # print(mid.melody[0]["notes"])


if __name__ == "__main__":
    main()
