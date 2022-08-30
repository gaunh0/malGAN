import torch
import numpy as np
from sklearn import linear_model, tree
from sklearn.neural_network import MLPClassifier
from torch import nn, optim
from torch.autograd.variable import Variable
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
from progress.bar import Bar
import os, shutil
from Script import genscript

loss = None
d_loss_arr = []
g_loss_arr = []
n_features = 128

class DiscriminatorNet(torch.nn.Module):
    """
    the discriminator that contains 2 hidden layer
    this networks works to decrease loss and learn what the adversarial examples are
    """

    def __init__(self, sigmoid):
        super(DiscriminatorNet, self).__init__()
        if sigmoid:
            self.hidden0 = nn.Sequential(nn.Linear(n_features, 256), nn.LeakyReLU(0.01))
            self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.01))
            #self.hidden2 = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(0.01))
            self.out = nn.Sequential(torch.nn.Linear(512, 1), nn.Sigmoid())

        else:
            self.hidden0 = nn.Sequential(nn.Linear(n_features, 256), nn.LeakyReLU(0.01))
            self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.01))
            #self.hidden2 = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(0.1))
            self.out = nn.Sequential(torch.nn.Linear(512, 1), torch.nn.Tanh())

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        #x = self.hidden2(x)
        x = self.out(x)
        return x


class GeneratorNet(torch.nn.Module):
    """
    a neural network with two hidden layers
    this network is used to create adversarial examples
    """

    def __init__(self, sigmoid, z_dimention):
        super(GeneratorNet, self).__init__()
        if sigmoid:
            self.hidden0 = nn.Sequential(
                nn.Linear(n_features + z_dimention, 256), nn.LeakyReLU(0.01)
            )
            self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.01))
            #self.hidden2 = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(0.1))
            self.out = nn.Sequential(nn.Linear(512, n_features), torch.nn.Sigmoid())

        else:
            self.hidden0 = nn.Sequential(
                nn.Linear(n_features + z_dimention, 256), nn.LeakyReLU(0.01)
            )
            self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.01))
            #self.hidden2 = nn.Sequential(nn.Linear(512, 512))
            self.out = nn.Sequential(nn.Linear(512, n_features), torch.nn.Tanh())

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        #x = self.hidden2(x)
        x = self.out(x)
        return x


def train_discriminator(
    optimizer, real_data, real_data_labels, fake_data, fake_data_labels
):
    """
    trains the discriminator or real data and fake data
    and computes a loss for each of them
    """
    global discriminator, generator
    optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = nn.MSELoss()(prediction_real, real_data_labels)
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = nn.MSELoss()(prediction_fake, fake_data_labels)
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data, fake_data_labels):
    """
    trains the gnerator giving the discriminator fake data and computes tje loss
    """
    global discriminator, loss, generator
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = nn.MSELoss()(prediction, fake_data_labels)
    error.backward()
    optimizer.step()
    return error


def load_data():
    """
    loads the data set
    :return:
    """
    data_set = "data1.npz"
    data = np.load(data_set)
    xmal, ymal, xben, yben = data["xmal"], data["ymal"], data["xben"], data["yben"]
    return (xmal, ymal), (xben, yben)


def train(
    num_epochs,
    blackbox,
    generator,
    d_optimizer,
    g_optimizer,
    train_tpr,
    test_tpr,
    sigmoid,
    isFirst=True,
):
    """
    trains MalGAN by creating a blackbox and feeding it the data then teaching a generator to make fake examples
    :param num_epochs:
    :param blackbox:
    :param generator:
    :param d_optimizer:
    :param g_optimizer:
    :param train_tpr:
    :param test_tpr:
    :param sigmoid:
    :param isFirst:
    :return:
    """
    global track_loss
    global g_loss_arr, d_loss_arr
    if isFirst:
        test_size = 0.2
    else:
        test_size = 0.5
    (mal, mal_label), (ben, ben_label) = load_data()

    x_train_mal, x_test_mal, y_train_mal, y_test_mal = train_test_split(
        mal, mal_label, test_size=test_size
    )

    # print("Malware train size: " + str(len(x_train_mal)))
    # print("Malware test size: " + str(len(x_test_mal)))
    x_train_ben, x_test_ben, y_train_ben, y_test_ben = train_test_split(
        ben, ben_label, test_size=test_size
    )
    # print("bengin train size: " + str(len(x_train_ben)))
    # print("bengin test size: " + str(len(x_test_ben)))

    (
        blackbox_x_train_mal,
        blackbox_y_train_mal,
        blackbox_x_train_ben,
        blackbox_y_train_ben,
    ) = (x_train_mal, y_train_mal, x_train_ben, y_train_ben)

    # train blackbox detecter
    if isFirst:
        blackbox.fit(np.concatenate([mal, ben]), np.concatenate([mal_label, ben_label]))

    ytrain_ben_blackbox = blackbox.predict(blackbox_x_train_ben)

    train_TPR = blackbox.score(blackbox_x_train_mal, blackbox_y_train_mal)
    test_TPR = blackbox.score(x_test_mal, y_test_mal)
    print("\n---TPR before MalGAN is trainning.")
    print("Train_TPR: {0}, Test_TPR: {1}\n".format(train_TPR, test_TPR))
    train_tpr.append(train_TPR)
    test_tpr.append(test_TPR)
    batch_size = 64
    print("Start training..")
    bar = Bar("Progress", max=num_epochs)
    # ****************
    # Create folder to save best model
    if not os.path.exists("Best_loss_model/"):
        os.makedirs("Best_loss_model/")
    # ****************
    for epoch in range(num_epochs):
        for step in range(x_train_mal.shape[0] // batch_size):
            d_loss_batches = []
            g_loss_batches = []

            #  generate batch of malware
            idm = np.random.randint(0, x_train_mal.shape[0], batch_size)
            if sigmoid:
                noise = np.random.uniform(0, 1, (batch_size, 20))
            else:
                noise = np.random.uniform(-1, 1, (batch_size, 20))
            xmal_batch = x_train_mal[idm]
            # generate batch of benign
            idb = np.random.randint(0, x_train_ben.shape[0], batch_size)
            xben_batch = x_train_ben[idb]
            yben_batch = ytrain_ben_blackbox[idb]

            # generate MALWARE examples
            combined = np.concatenate([xmal_batch, noise], axis=1)
            fake_mal_data = generator(torch.from_numpy(combined).float())

            # change the labels based on which activation function is being used
            if sigmoid:
                ymal_batch = blackbox.predict(
                    np.ones(fake_mal_data.shape)
                    * (np.asarray(fake_mal_data.detach()) > 0.5)
                )
            else:
                ymal_batch = blackbox.predict(
                    np.ones(fake_mal_data.shape)
                    * (np.asarray(fake_mal_data.detach()) > 0)
                )

            xben_batch = torch.from_numpy(xben_batch).float()
            yben_batch = torch.from_numpy(yben_batch).float()
            yben_batch = yben_batch.unsqueeze(1)
            ymal_batch = torch.from_numpy(ymal_batch).float()
            ymal_batch = ymal_batch.unsqueeze(1)

            # train discriminator
            d_loss, d_pred_real, d_pred_fake = train_discriminator(
                d_optimizer, xben_batch, yben_batch, fake_mal_data, ymal_batch
            )
            d_loss_batches.append(d_loss.item())

            # train generator on noise
            g_loss = train_generator(
                g_optimizer, torch.from_numpy(xmal_batch).float(), yben_batch
            )
            g_loss_batches.append(g_loss.item())

        # add loss to array
        d_loss_arr.append(min(d_loss_batches))
        g_loss_arr.append(min(g_loss_batches))
        # ******************
        # TRACKING LOSS FOR DISCRIMINATOR
        epoch_loss = min(d_loss_batches)
        if epoch == 0:
            track_loss = epoch_loss
            # torch.save(generator.state_dict(),'Best_loss_model/gen_epoch_{}_loss_{}_isFirst_{}_.pth'.format(epoch, track_loss, isFirst))
        else:
            if track_loss < epoch_loss:
                track_loss = epoch_loss
        # print('Track_loss: {}, epoch_d_loss: {}'.format(track_loss,epoch_loss))
        if not os.listdir("Best_loss_model/"):
            torch.save(
                generator.state_dict(),
                "Best_loss_model/gen_epoch_{}_loss_{}_isFirst_{}_.pth".format(
                    epoch, track_loss, isFirst
                ),
            )
        else:
            ##remove old weights and save new weights (only-1-weights)
            for files in os.listdir("Best_loss_model/"):
                path = os.path.join("Best_loss_model/", files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)
            torch.save(
                generator.state_dict(),
                "Best_loss_model/gen_epoch_{}_loss_{}_isFirst_{}_.pth".format(
                    epoch, track_loss, isFirst
                ),
            )

        # ******************
        # train true positive rate
        if sigmoid:
            noise = np.random.uniform(0, 1, (x_train_mal.shape[0], 20))
        else:
            noise = np.random.uniform(-1, 1, (x_train_mal.shape[0], 20))

        combined = np.concatenate([x_train_mal, noise], axis=1)
        gen_examples = generator(torch.from_numpy(combined).float())

        if sigmoid:
            train_TPR = blackbox.score(
                np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0.5),
                y_train_mal,
            )
        else:
            train_TPR = blackbox.score(
                np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0),
                y_train_mal,
            )

        # test true positive rate
        if sigmoid:
            noise = np.random.uniform(0, 1, (x_test_mal.shape[0], 20))
        else:
            noise = np.random.uniform(-1, 1, (x_test_mal.shape[0], 20))

        combined = np.concatenate([x_test_mal, noise], axis=1)
        gen_examples = generator(torch.from_numpy(combined).float())

        if sigmoid:
            test_TPR = blackbox.score(
                np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0.5),
                y_test_mal,
            )
        else:
            test_TPR = blackbox.score(
                np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0),
                y_test_mal,
            )

        # print(epoch, " ", train_TPR, " ", test_TPR)

        train_tpr.append(train_TPR)
        test_tpr.append(test_TPR)
        bar.next()
    bar.finish()
    
    
def evaluate(z_dim, sigmoid, blackbox, num_examples= 10, include_data_labels = True):
    (mal, mal_lables), _ = load_data()

    gen_weights_model_path = os.path.join('Best_loss_model/',os.listdir('Best_loss_model/')[0])

    gen_model = GeneratorNet(sigmoid=sigmoid, z_dimention=z_dim)
    gen_model.load_state_dict(torch.load(gen_weights_model_path))
    gen_model.eval()

    mal_examples = mal[:num_examples]
    if sigmoid:
            noise = np.random.uniform(0, 1, (num_examples, 20))
    else:
        noise = np.random.uniform(-1, 1, (num_examples, 20))

    combined = np.concatenate([mal_examples, noise], axis=1)#*******
    gen_examples = gen_model(torch.from_numpy(combined).float())

    if include_data_labels:
        y = mal_lables[:num_examples]
        if sigmoid:
            eval_score = blackbox.score(
                np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0.5), y)
        else:
            eval_score = blackbox.score(
                np.ones(gen_examples.shape) * (np.asarray(gen_examples.detach()) > 0), y)
    else:
        print("\nLabels: 1: Malware, 0: Benin")

        _class = blackbox.predict(gen_examples.detach())
        _class_proba = blackbox.predict_proba(gen_examples.detach())
        print("Predicted: ",_class)
        print("Probability: ",_class_proba)
        y_real = np.ones(num_examples)
        print("Y_true: ",y_real)

        diff = _class - y_real
        eval_score = (num_examples- np.count_nonzero(diff))/num_examples

    print("Black_box_score: {}".format(eval_score),"\n")
    array = gen_examples.flatten()
    genscript.generate(array)

def retrain(blackbox, generator, sigmoid):
    """
    retrain the blackbox after the malgan has beeen trained
    :param blackbox:
    :param generator:
    :param sigmoid:
    :return:
    """
    (mal, mal_label), (ben, ben_label) = load_data()
    x_train_mal, x_test_mal, y_train_mal, y_test_mal = train_test_split(
        mal, mal_label, test_size=0.20
    )
    x_train_ben, x_test_ben, y_train_ben, y_test_ben = train_test_split(
        ben, ben_label, test_size=0.20
    )

    # Generate Train Adversarial Examples
    if sigmoid:
        noise = np.random.uniform(0, 1, (x_train_mal.shape[0], 20))
    else:
        noise = np.random.uniform(-1, 1, (x_train_mal.shape[0], 20))
    combined = np.concatenate([x_train_mal, noise], axis=1)
    gen_examples = generator(torch.from_numpy(combined).float())

    gen_examples_np = np.asarray(gen_examples.detach())

    blackbox.fit(
        np.concatenate([x_train_mal, x_train_ben, gen_examples_np]),
        np.concatenate([y_train_mal, y_train_ben, np.zeros(gen_examples.shape[0])]),
    )

    # training true positive rate
    train_TPR = blackbox.score(np.asarray(gen_examples.detach()), y_train_mal)

    # test true positive rate
    if sigmoid:
        noise = np.random.uniform(0, 1, (x_test_mal.shape[0], 20))
    else:
        noise = np.random.uniform(-1, 1, (x_test_mal.shape[0], 20))

    combined = np.concatenate([x_test_mal, noise], axis=1)
    gen_examples = generator(torch.from_numpy(combined).float())

    if sigmoid:
        gen_examples = np.ones(gen_examples.shape) * (
            np.asarray(gen_examples.detach()) > 0.5
        )
    else:
        gen_examples = np.ones(gen_examples.shape) * (
            np.asarray(gen_examples.detach()) > 0
        )

    test_TPR = blackbox.score(gen_examples, y_test_mal)

    print(
        "\n---TPR after the black-box detector is retrained(Before Retraining MalGAN)."
    )
    print("\nTrain_TPR: {0}, Test_TPR: {1}".format(train_TPR, test_TPR))


def arument_parser():
    parser = argparse.ArgumentParser(description="Malgan parameter.")

    parser.add_argument(
        "z",
        metavar="Z",
        type=int,
        help="Noise dimension",
    )

    parser.add_argument(
        "train_epoch",
        metavar="train_epoch",
        type=int,
        help="Train epoch",
    )
    parser.add_argument(
        "retrain_epoch",
        metavar="retrain_epoch",
        type=int,
        help="Retrain epoch",
    )
    parser.add_argument(
        "--blackbox",
        dest="blackbox",
        help="Blackbox model:\n RF: Random Forest (default)\n LG: Logistic Regression\n DT: Decision Tree",
    )

    args = parser.parse_args()
    if args.blackbox == None:
        args.blackbox = "RF"
    return args


def main():
    arg = arument_parser()
    z_dimention = arg.z
    train_epoch = arg.train_epoch
    retrain_epoch = arg.retrain_epoch
    global discriminator, generator, loss
    sigmoid = True
    # initialize the discriminator and generator
    discriminator = DiscriminatorNet(sigmoid)
    generator = GeneratorNet(sigmoid, z_dimention)

    print("**********Discriminator structure************")
    print(discriminator)
    print("*************Generator structure*************")
    print(generator)

    ## DIFFERENT BLACKBOX OPTIONS TO TEST
    ## COMMENT OUT THE ONES THAT ARE NOT BEING TESTED
    if arg.blackbox == "RF":
        blackbox = RandomForestClassifier(
            n_estimators=101, max_depth=10, random_state=1
        )
    elif arg.blackbox == "LG":
        blackbox = linear_model.LogisticRegression()
    else:
        blackbox = tree.DecisionTreeRegressor()

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

    # arrays for plotting
    TRAIN_TPR = []
    TEST_TPR = []
    POST_TRAIN_TPR = []
    POST_TEST_TPR = []

    # train the gan on examples
    train(
        train_epoch,
        blackbox,
        generator,
        d_optimizer,
        g_optimizer,
        TRAIN_TPR,
        TEST_TPR,
        sigmoid,
        isFirst=True,
    )

    # retrain the blackbox
    retrain(blackbox, generator, sigmoid)

    # run malgan again
    train(
        retrain_epoch,
        blackbox,
        generator,
        d_optimizer,
        g_optimizer,
        POST_TRAIN_TPR,
        POST_TEST_TPR,
        sigmoid,
        isFirst=False,
    )

    print(
        "\nTPR before: {0}, TPR after: {1}".format(
            TEST_TPR[len(TEST_TPR) - 1], POST_TEST_TPR[len(POST_TEST_TPR) - 1]
        )
    )
    #evaluate(z_dimention, sigmoid, blackbox, num_examples=1, include_data_labels=False)

    # plot data
    plt.plot(
        range(len(d_loss_arr)), d_loss_arr, label="Discriminator loss", color="blue"
    )
    plt.plot(range(len(g_loss_arr)), g_loss_arr, label="Generator Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Min Loss Value per Epoch")
    plt.title("Graph of losses over time")
    plt.legend()
    plt.show()

    plt.plot(range(len(TRAIN_TPR)), TRAIN_TPR, label="Training TPR", color="blue")
    plt.plot(range(len(TEST_TPR)), TEST_TPR, label="Testing TPR", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("TPR rate")
    plt.legend()
    plt.title("True Positive Rate of Blackbox before retraining MalGAN")
    plt.show()

    plt.plot(
        range(len(POST_TRAIN_TPR)), POST_TRAIN_TPR, label="Training TPR", color="blue"
    )
    plt.plot(range(len(POST_TEST_TPR)), POST_TEST_TPR, label="Testing TPR", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("TPR rate")
    plt.legend()
    plt.title("True Positive Rate of Blackbox after retraining MalGAN")
    plt.show()


if __name__ == "__main__":
    main()

