
import matplotlib.pyplot as plt

def plot_result(label_text, list_x, list_y, list_y_filtered, raw_d1, filtered_d1):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(list_x, list_y)
    ax[0, 0].set_title(label_text + ":::raw y")
    ax[0, 1].plot(list_x, list_y_filtered[0])
    ax[0, 1].set_title(label_text + ":::filtered y")
    ax[1, 0].plot(list_x, raw_d1)
    ax[1, 0].set_title(label_text + ":::raw d1")
    ax[1, 1].plot(list_x, filtered_d1)
    ax[1, 1].set_title(label_text + ":::filtered d1")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid()
    plt.legend()
    plt.show()

def plot_result_with_level_prediction(label_text, list_x, list_y, list_possible_predictions):
    plt.plot(list_x, list_y)
    for optimal_point in list_possible_predictions:
        plt.plot(optimal_point[0], optimal_point[1], marker="x", markersize=4, markeredgecolor="red", markerfacecolor="green")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.grid()
    plt.legend()
    plt.show()

def save_result(label_text, x, list_y, list_y_filtered, raw_d1, filtered_d1):
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0, 0].plot(x, list_y)
    ax[0, 0].set_title(label_text + ":::raw y")
    ax[0, 1].plot(x, list_y_filtered[0])
    ax[0, 1].set_title(label_text + ":::filtered y")
    ax[1, 0].plot(x, raw_d1)
    ax[1, 0].set_title(label_text + ":::raw d1")
    ax[1, 1].plot(x, filtered_d1)
    ax[1, 1].set_title(label_text + ":::filtered d1")
    plt.savefig(label_text + '.png')

