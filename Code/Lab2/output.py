from matplotlib import pyplot as plt

TXT_PATH = "Texts/output1.txt"


def set_txt_name(name):
    global TXT_PATH
    TXT_PATH = "Texts/" + name + ".txt"


def output(model_name, history):
    with open(TXT_PATH, "a") as f:
        f.write(model_name + "\n")
        f.write("Final Loss: " + str(history.history['loss'][-1]) + "\n")
        f.write("Final Test Loss: " + str(history.history['val_loss'][-1]) + "\n")
        f.write("\n")
    print(model_name, "finished training")

    plt.title(model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Test loss')
    plt.legend()
    plt.show()


def test_start(test_object_string):
    with open(TXT_PATH, "a") as f:
        f.write('____________________________________________________________________\n')
        f.write(test_object_string + '\n')
        f.write('--------------------------------------------------------------------\n')


def test_output(model_name, history):
    with open(TXT_PATH, "a") as f:
        f.write(str(model_name) + '\n')
        f.write("Final Loss: " + str(history.history['loss'][-1]) + "\n")
        f.write("Final Test Loss: " + str(history.history['val_loss'][-1]) + "\n")
        f.write("\n")


def test_parameters_output(value, history, model_name):
    with open(TXT_PATH, "a") as f:
        f.write('____________________________________________________________________\n')
        f.write(str(value) + '\n')
        f.write('--------------------------------------------------------------------\n')
        f.write("Final Loss: " + str(history.history['loss'][-1]) + "\n")
        f.write("Final Test Loss: " + str(history.history['val_loss'][-1]) + "\n")
        f.write("\n")
    print(model_name, "finished training")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Test loss')
    plt.legend()
    plt.show()
