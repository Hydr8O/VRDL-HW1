import matplotlib.pyplot as plt


def plot_losses_accuracies(losses, accuracies, num_epochs):
    plt.plot(range(num_epochs), losses['train'])
    plt.plot(range(num_epochs), losses['val'])
    plt.title('Loss statistics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train loss', 'Validation loss'])
    plt.savefig('losses.png')
    plt.close()

    plt.plot(range(num_epochs), accuracies['train'])
    plt.plot(range(num_epochs), accuracies['val'])
    plt.title('Accuracy statistics')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train acc', 'Validation acc'])
    plt.savefig('accuracies.png')
    plt.close()


def plot_learning_rate(learning_rate, num_epochs):
    plt.plot(range(num_epochs), learning_rate)
    plt.title('Learning rate over time')
    plt.xlabel('Epochs')
    plt.ylabel('Learning rate')
    plt.savefig('lr.png')
    plt.close()
