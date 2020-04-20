import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

def visualize(nnet, train_loader, dir, args):
    ch = args.channels[0]
    # turn off gradients and other aspects of training
    nnet.eval()
    with torch.no_grad():

        n_items = len(train_loader.dataset)
        r = 3 if n_items >= 3 else n_items
        for i in range(r):
            n = np.random.randint(0, n_items)
            X, T, filename = train_loader.dataset[n]
            X = X[args.channels,...]
            dims = X.shape
            print(dims, T, filename)
            o_filename = f'{dir}{i}_original'
            save_1(X[ch:ch+1,...], o_filename, f'T: {T}, f: {filename}_o, [{ch}]')

            m_filename = f'{dir}{i}_modified'
            if 'lin' in args.model:
                X = X.reshape(1, np.product(X.shape[0:]))
                Y = nnet(X).detach().numpy().reshape(dims)
            elif 'cnn' in args.model:
                X = torch.from_numpy(np.expand_dims(X, axis=0)).float()
                Y = nnet(X).detach().numpy()[0, ...]
            elif 'vae' in args.model:
                X = torch.from_numpy(np.expand_dims(X, axis=0)).float()
                Y = nnet(X)[0].detach().numpy()[0, ...]
            save_1(Y[ch:ch+1,...], m_filename, f'T: {T}, f: {filename}_m, [{ch}]')


def save_1(X, filename, title):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    im = ax.imshow(X.reshape(X.shape[-2], X.shape[-1]),
                          interpolation='nearest', cmap='viridis')

    ax.set_title(title, fontsize=4)
    fig.colorbar(im)
    fig.savefig(filename + '.pdf', dpi=600, bbox_inches='tight');
