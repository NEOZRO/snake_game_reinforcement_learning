import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import pandas as pd

def plot_live_results(x):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.style.use("seaborn-dark")


    if x is not None:

        plt.plot(x)

        # MOVING AVERAGE
        if len(x) >= 20:
            x_ma = pd.Series(x).rolling(window=20).mean()
            plt.plot(x_ma)
        else:
            plt.plot(x)

        # plt.show()  # Display the plot

        fig.canvas.draw()

        # convert canvas to image
        img_grafico = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                         sep='')
        img_grafico = img_grafico.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img_grafico = cv2.cvtColor(img_grafico, cv2.COLOR_RGB2BGR)

        cv2.imshow("plot", img_grafico)

        plt.close()

    else:
        pass


def plot_live_results_tron(x, y):

    plt.style.use("dark_background")

    for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        plt.rcParams[param] = '0.9'

    for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        plt.rcParams[param] = '#212946'

    colors = [
        '#00ff41',  # green
        '#FE53BB']  # pink

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.grid(color='#2A3459')
    if x is not None or y is not None:

        plt.plot(x, color=colors[1])
        plt.plot(y, color=colors[0])

        n_shades = 4
        diff_linewidth = 1.9
        alpha_value = 0.3/n_shades
        for n in range(1, n_shades+1):

            plt.plot(x, color=colors[1], linewidth=2+(diff_linewidth*n), alpha=alpha_value)
            plt.plot(y, color=colors[0], linewidth=2+(diff_linewidth*n), alpha=alpha_value)

            # df.plot(marker='o',
            #         linewidth=2+(diff_linewidth*n),
            #         alpha=alpha_value,
            #         legend=False,
            #         ax=ax,
            #         color=colors)
        ax.grid(color='#2A3459')

        fig.canvas.draw()

        # convert canvas to image
        img_grafico = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                    sep='')
        img_grafico = img_grafico.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img_grafico = cv2.cvtColor(img_grafico, cv2.COLOR_RGB2BGR)

        cv2.imshow("plot", img_grafico)

        plt.close()

    else:
        pass