import matplotlib.pyplot as plt
plt.rc('font', family='serif')

def save_1(X, filename, title):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    im = ax.imshow(X.reshape(X.shape[-2], X.shape[-1]),
                          interpolation='nearest', cmap='viridis')

    ax.set_title(title)
    fig.colorbar(im)
    fig.savefig(filename + '.pdf', dpi=600, bbox_inches='tight');
