from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

kl_w = DEFAULT_KL_WEIGHT
classifier_w = DEFAULT_CLASSIFIER_WEIGHT
reconstruction_w = DEFAULT_RECONSTRUCTION_WEIGHT


class CsvDataset(torch_data.Dataset):
    def __init__(self, path):
        self.data = []

        dataframe = pd.read_csv(path)
        column_names = dataframe.columns.tolist()

        for features, label in tqdm(zip(dataframe[column_names[1]], dataframe[column_names[2]]), desc="loading data", total=len(dataframe)):

            features = features.replace('[', '')
            features = features.replace(']', '')
            features = features.split(',')

            for i, value in enumerate(features):
                features[i] = float(value)

            self.data.append((torch.tensor(features, dtype=torch.float32), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        self.activation = DEFAULT_ACTIVATION

        self.linear1 = nn.Linear(DEFAULT_VAE_INPUT_SIZE, DEFAULT_VAE_HIDDEN1_SIZE, bias=False)
        self.linear2 = nn.Linear(DEFAULT_VAE_HIDDEN1_SIZE, DEFAULT_VAE_HIDDEN2_SIZE, bias=False)
        self.linear3 = nn.Linear(DEFAULT_VAE_HIDDEN2_SIZE, DEFAULT_VAE_HIDDEN3_SIZE, bias=False)
        self.linear4 = nn.Linear(DEFAULT_VAE_HIDDEN3_SIZE, latent_dims, bias=False)
        self.linear5 = nn.Linear(DEFAULT_VAE_HIDDEN3_SIZE, latent_dims)

        self.batch1 = nn.BatchNorm1d(DEFAULT_VAE_HIDDEN1_SIZE)
        self.batch2 = nn.BatchNorm1d(DEFAULT_VAE_HIDDEN2_SIZE)
        self.batch3 = nn.BatchNorm1d(DEFAULT_VAE_HIDDEN3_SIZE)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

        if device == "cuda":
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()

    def forward(self, x):
        x = x.to(device)
        x = self.activation(self.batch1(self.linear1(x)))
        x = self.activation(self.batch2(self.linear2(x)))
        x = self.activation(self.batch3(self.linear3(x)))

        mu = self.linear4(x)
        sigma = torch.exp(self.linear5(x))

        # reparameterization
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()

        self.activation = DEFAULT_ACTIVATION
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, DEFAULT_VAE_HIDDEN3_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_VAE_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_VAE_HIDDEN3_SIZE, DEFAULT_VAE_HIDDEN2_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_VAE_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_VAE_HIDDEN2_SIZE, DEFAULT_VAE_HIDDEN1_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_VAE_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_VAE_HIDDEN1_SIZE, DEFAULT_VAE_INPUT_SIZE, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        decoded_x = self.decoder(x)

        return decoded_x


class VariationalAutoencoder(nn.Module):

    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

        self.latent_dims = latent_dims
        self.vector_target = []

    def forward(self, x):
        x = x.to(device)
        encoded_x = self.encoder(x)

        return self.decoder(encoded_x)


class Classifier(nn.Module):
    def __init__(self, latent_dims):
        super(Classifier, self).__init__()

        self.activation = DEFAULT_ACTIVATION
        self.classifier = nn.Sequential(
            nn.Linear(latent_dims, DEFAULT_CLASSIFIER_HIDDEN1_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_CLASSIFIER_HIDDEN1_SIZE),
            self.activation,

            nn.Linear(DEFAULT_CLASSIFIER_HIDDEN1_SIZE, DEFAULT_CLASSIFIER_HIDDEN2_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_CLASSIFIER_HIDDEN2_SIZE),
            self.activation,

            nn.Linear(DEFAULT_CLASSIFIER_HIDDEN2_SIZE, DEFAULT_CLASSIFIER_HIDDEN3_SIZE, bias=False),
            nn.BatchNorm1d(DEFAULT_CLASSIFIER_HIDDEN3_SIZE),
            self.activation,

            nn.Linear(DEFAULT_CLASSIFIER_HIDDEN3_SIZE, DEFAULT_CLASSIFIER_OUTPUT_SIZE, bias=False),
        )
        self.latent_dims = latent_dims

    def forward(self, x):
        decoded_x = self.classifier(x)

        return decoded_x


def train(vae: VariationalAutoencoder, classifier, dataset, epochs: int):
    params = list(vae.parameters()) + list(classifier.parameters())
    opt = torch.optim.Adam(params, lr=DEFAULT_LEARNING_RATE)

    # Define the proportions for each subset
    train_ratio = 0.8
    val_ratio = 0.1

    # Calculate the lengths of each subset
    train_len = int(len(dataset) * train_ratio)
    val_len = int(len(dataset) * val_ratio)
    test_len = int(len(dataset) - (train_len + val_len))

    train_subset, val_subset, test_subset = random_split(dataset, [train_len, val_len, test_len])

    training_loader = torch_data.DataLoader(train_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=True)
    validation_loader = torch_data.DataLoader(val_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    test_loader = torch_data.DataLoader(test_subset, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    epoch_losses = []
    val_losses = []
    epochs_entropy = []
    epochs_precision = []
    epochs_accuracy = []

    for epoch in range(epochs):

        epoch_loss, entropy_train, precision_train, accuracy_train = train_epoch(vae, classifier, training_loader, opt)
        val_loss, entropy_val, precision_val, accuracy_val = val_epoch(vae, classifier, validation_loader)

        entropy = (entropy_train + entropy_val) / 2
        precision = (precision_train + precision_val) / 2
        accuracy = (accuracy_train + accuracy_val) / 2

        epoch_losses.append(epoch_loss)
        val_losses.append(val_loss)
        epochs_entropy.append(entropy)
        epochs_precision.append(precision)
        epochs_accuracy.append(accuracy)

        print(f"epoch: {epoch+1}/{epochs}, epoch loss = {epoch_loss}\tval loss = {val_loss}\tentropy = {entropy}"
              f"\tprecision = {precision}\taccuracy = {accuracy}\n")
        time.sleep(0.1)

    test_loss, entropy, precision, accuracy = test_epoch(vae, classifier, test_loader)
    print(f"test loss = {test_loss}\tentropy = {entropy}\tprecision = {precision}\taccuracy = {accuracy}\n")

    plt.plot(range(epochs), epoch_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    # plt.plot(range(epochs), epochs_entropy, label='Entropy')
    plt.plot(range(epochs), epochs_precision, label='Precision')
    # plt.plot(range(epochs), epochs_accuracy, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss at Different Epochs')
    plt.legend(loc='upper center')
    plt.savefig('files/epoch_losses.png')
    plt.show()

    return vae


def train_epoch(vae: VariationalAutoencoder, classifier, data, opt):
    # Set train mode for both the encoder and the decoder
    vae.train()
    classifier.train()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    epoch_loss = 0.0
    entropy = 0.0
    precision = 0.0
    accuracy = 0.0

    for x, y in tqdm(data, desc="training epoch", unit="batch"):

        x = x.to(device)
        # y = y.to(device)

        opt.zero_grad()

        encoded_x = vae.encoder(x)
        x_hat = vae(x)
        y_hat = classifier(encoded_x)

        vae_loss = ((x - x_hat) ** 2).sum()
        classifier_loss = classifier_criterion(y_hat, y)

        loss = (kl_w * vae.encoder.kl) + (classifier_w * classifier_loss) + (reconstruction_w * vae_loss)

        _, y_pred = torch.max(y_hat, 1)
        p = precision_score(y.numpy(), y_pred.numpy(), average='micro')
        acc = accuracy_score(y.numpy(), y_pred.numpy())

        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        entropy += vae_loss.item()
        precision += p
        accuracy += acc

    len_data = round(len(data.dataset) / DEFAULT_BATCH_SIZE)

    epoch_loss /= len_data
    entropy /= len_data
    precision = precision / len_data
    accuracy /= len_data

    return epoch_loss, entropy, precision, accuracy


def val_epoch(vae: VariationalAutoencoder, classifier, data):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    classifier.eval()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    val_loss = 0.0
    entropy = 0.0
    precision = 0.0
    accuracy = 0.0

    with torch.no_grad(): # No need to track the gradients
        for x, y in tqdm(data, desc="validating epoch", unit="batch"):

            x = x.to(device)
            # y = y.to(device)

            encoded_x = vae.encoder(x)
            x_hat = vae(x)
            y_hat = classifier(encoded_x)

            vae_loss = ((x - x_hat)**2).sum()
            classifier_loss = classifier_criterion(y_hat, y)

            loss = (kl_w * vae.encoder.kl) + (classifier_w * classifier_loss) + (reconstruction_w * vae_loss)

            _, y_pred = torch.max(y_hat, 1)
            p = precision_score(y.numpy(), y_pred.numpy(), average='micro')
            acc = accuracy_score(y.numpy(), y_pred.numpy())

            val_loss += loss.item()
            entropy += vae_loss.item()
            precision += p
            accuracy += acc

    len_data = round(len(data.dataset) / DEFAULT_BATCH_SIZE)

    val_loss /= len_data
    entropy /= len_data
    precision = precision / len_data
    accuracy /= len_data

    return val_loss, entropy, precision, accuracy


def test_epoch(vae: VariationalAutoencoder, classifier, data):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    classifier.eval()

    classifier_criterion = DEFAULT_CLASSIFIER_CRITERION
    test_loss = 0.0
    entropy = 0.0
    precision = 0.0
    accuracy = 0.0

    with torch.no_grad(): # No need to track the gradients
        for x, y in tqdm(data, desc="testing epochs", unit="batch"):

            x = x.to(device)
            # y = y.to(device)

            encoded_x = vae.encoder(x)
            x_hat = vae(x)
            y_hat = classifier(encoded_x)

            vae_loss = ((x - x_hat)**2).sum()
            classifier_loss = classifier_criterion(y_hat, y)

            loss = (kl_w * vae.encoder.kl) + (classifier_w * classifier_loss) + (reconstruction_w * vae_loss)

            _, y_pred = torch.max(y_hat, 1)
            p = precision_score(y.numpy(), y_pred.numpy(), average='micro')
            acc = accuracy_score(y.numpy(), y_pred.numpy())

            test_loss += loss.item()
            entropy = vae_loss.item()
            precision += p
            accuracy += acc

            # print(f"y: {y}")
            # print(f"y_hat: {y_pred}")

    len_data = round(len(data.dataset) / DEFAULT_BATCH_SIZE)

    test_loss /= len_data
    entropy /= len_data
    precision /= len_data
    accuracy /= len_data

    return test_loss, entropy, precision, accuracy


def centroids(vae, data):
    latent_df = latent_to_df(vae, data)
    x = to_array(latent_df['points'])

    kmeans = KMedoids(n_clusters=NO_MAJ_KEYS, init="k-medoids++", max_iter=300).fit(x)

    return torch.FloatTensor(kmeans.cluster_centers_)


def randn_latent_walk(latent_dims, num_steps, step_size, v0):
    v0 = v0.reshape(1, latent_dims)

    steps = torch.randn(num_steps, latent_dims)
    rand_walk = torch.zeros(num_steps + 1, latent_dims)
    rand_walk[0] = v0

    for i in range(num_steps):
        rand_walk[i + 1] = rand_walk[i] + steps[i] * step_size

    rand_walk = torch.FloatTensor(rand_walk)
    return rand_walk


# def find_target_vectors(vae, classifier):
#
#     print("finding target vectors...")
#     v0 = torch.randn(1, vae.latent_dims)
#
#     for i in range(NO_MAJ_KEYS):
#         vector_target = find_target(classifier, v0, target=i)
#         vae.vector_target += [vector_target]
#
#
# def find_target(classifier, v0, target: int = 0, f_step_size: float = 0.1, max_steps: int = 1000, w_step_size: int = 1e-9, walk_size: int = 10):
#     vi = v0
#     i = 0
#
#     while not is_on_target(classifier, vi, target, walk_size, w_step_size):
#         step = torch.randn(1, classifier.latent_dims) * f_step_size
#         vi = vi + step
#
#         i += 1
#         if i > max_steps:
#             vi = v0
#             i = 0
#
#     return vi
#
#
# def is_on_target(classifier, vector, target, walk_size, step_size):
#     classifier.eval()
#     vector_walk = [vector[0].numpy()]
#
#     for _ in range(walk_size):
#         step = torch.randn(1, classifier.latent_dims) * step_size
#         vector += step
#
#         vector_walk.append(vector[0].numpy())
#
#     vector_walk = torch.FloatTensor(vector_walk)
#
#     label_pred = classifier(vector_walk)
#     _, label_pred = torch.max(label_pred, 1)
#
#     for label in label_pred:
#
#         if int(label) != target:
#             return False
#
#     return True


def to_array(df: pd.DataFrame) -> np.array:

    arr = []
    for item in df:
        arr += [item]

    return np.array(arr)


def latent_to_df(autoencoder: VariationalAutoencoder, data) -> pd.DataFrame:
    latent_df = pd.DataFrame()
    autoencoder.encoder.eval()

    data_loader = torch_data.DataLoader(data, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    latent_points = []
    latent_labels = []
    for file_batch, label_batch in tqdm(data_loader, desc="plotting batches", unit="batch", total=len(data_loader)):
        encoded_batch = autoencoder.encoder(file_batch)

        for encoded_file, label in zip(encoded_batch, label_batch):
            encoded_file = encoded_file.detach().numpy()

            latent_points += [encoded_file]
            latent_labels += [label.numpy()]

    latent_df["points"] = latent_points
    latent_df["labels"] = latent_labels

    return latent_df


def plot2d_latent(vae, data, show_plot: bool = True):

    from sklearn.manifold import TSNE

    latent_df = latent_to_df(vae, data)
    fig, ax = plt.subplots()

    if vae.latent_dims > 2:
        print(f"applying TSNE transformation...")

        points = to_array(latent_df["points"])

        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(points)

        x = tsne_results[:, 0].tolist()
        y = tsne_results[:, 1].tolist()

    else:
        x = []
        y = []

        for point in latent_df["points"]:
            x += [point[0]]
            y += [point[1]]

    x_centroid = []
    y_centroid = []
    for vector in vae.vector_target:
        x_centroid += [vector[0]]
        y_centroid += [vector[1]]

    ax.scatter(x, y, c=latent_df["labels"])
    ax.scatter(x_centroid, y_centroid, marker='*', c='black', s=100)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)

    if show_plot:
        fig.savefig('files/latent_space.png')
        plt.show()

    return fig, ax


def plot2d_walk(walk, ax_latent):

    x = []
    y = []
    for step in walk:
        x += [step[0]]
        y += [step[1]]

    ax_latent.plot(x, y, marker='o', c='black', linestyle='-')

    return ax_latent


def main():
    # Hyperparameters
    latent_dims = DEFAULT_LATENT_SIZE
    n_epochs = DEFAULT_N_EPOCHS

    dataset = CsvDataset("data/preprocessed_data/preprocessed_data.csv")

    vae = VariationalAutoencoder(latent_dims).to(device)  # GPU
    classifier = Classifier(latent_dims).to(device)

    vae = train(vae, classifier, dataset, n_epochs)
    vae.vector_target = centroids(vae, dataset) # K-means

    plot2d_latent(vae, dataset)

    torch.save(vae.state_dict(), 'files/music_vae.pt')
    torch.save(classifier.state_dict(), 'files/music_classifier.pt')

    torch.save(vae.vector_target, 'files/vector_targets.pt')


if __name__ == '__main__':
    main()
