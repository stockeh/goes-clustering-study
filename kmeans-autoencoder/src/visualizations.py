import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

def visualize(nnet, train_loader, args):
    ch = args.channels[0]
    # turn off gradients and other aspects of training
    nnet.eval()
    with torch.no_grad():

        n_items = len(train_loader.dataset)
        r = 3 if n_items >= 3 else n_items
        for i in range(r):
            n = np.random.randint(0, n_items)
            o_filename = f'../media/{str(i)}_original'
            X = train_loader.dataset[n][args.channels,...]
            dims = X.shape
            save_1(X[ch:ch+1,...], o_filename, f'X, f: {n}, [{ch}]')

            m_filename = f'../media/{str(i)}_modified'
            if 'lin' in args.model:
                X = X.reshape(1, np.product(X.shape[0:]))
                Y = nnet(X).detach().numpy().reshape(dims)
            elif 'cnn' in args.model:
                X = torch.from_numpy(np.expand_dims(X, axis=0)).float()
                Y = nnet(X).detach().numpy()[0, ...]
            elif 'vae' in args.model:
                X = torch.from_numpy(np.expand_dims(X, axis=0)).float()
                Y = nnet(X)[0].detach().numpy()[0, ...]
            save_1(Y[ch:ch+1,...], m_filename, f'Y, f: {n}, [{ch}]')


def save_1(X, filename, title):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    im = ax.imshow(X.reshape(X.shape[-2], X.shape[-1]),
                          interpolation='nearest', cmap='viridis')

    ax.set_title(title)
    fig.colorbar(im)
    fig.savefig(filename + '.pdf', dpi=600, bbox_inches='tight');
