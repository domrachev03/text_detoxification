import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import wordcloud
from PIL import Image


def plot_histograms(
    df: pd.DataFrame,
    save=False,
    fname_hist='image/raw_histogram.png'
):
    fig, ax = plt.subplots(2, 2, figsize=(2*4, 2*3))
    ax = ax.ravel()

    i = 0
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        ax[i].set_xlabel("Value")
        ax[i].set_ylabel("no. of entries")
        ax[i].set_title(col.replace('_', ' ').capitalize())
        ax[i].hist(df[col], 20)
        i += 1

    plt.tight_layout()
    if save:
        plt.savefig(fname_hist)
    else:
        plt.show()
    plt.clf()


def plot_dist_n_length(
    df: pd.DataFrame,
    save=False,
    fname='image/length_distribution.png'
):
    fig, ax = plt.subplots(2, 1, figsize=((2*4, 2*3)))

    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("no. of entries")
    ax[0].set_title("Toxicity difference")
    ax[0].hist(df['ref_tox']-df["trn_tox"], 20)

    ax[1].set_xlabel("Length")
    ax[1].set_ylabel("no. of entries")
    ax[1].set_title("Toxicity difference")
    for corpus in ['reference', 'translation']:
        length = np.array([len(s) for s in df[corpus]])
        elements, repears = np.unique(length, return_counts=True)
        ax[1].plot(elements[:200], repears[:200], label=corpus)
    plt.tight_layout()
    plt.legend()

    if save:
        fig.savefig(fname)
    else:
        plt.show()
    plt.clf()


def visualize_via_wordcloud(
    ref: pd.DataFrame,
    trans: pd.DataFrame,
    save: bool = False,
    fname: str = 'wordcloud.png'
) -> Image:
    general_mask = np.array(Image.open('image/devil_and_angel.png'))
    h, w = general_mask.shape[0:2]
    mask_ref, mask_trs = general_mask[:, :w//2], general_mask[:, w//2:] 

    stopwords = set(wordcloud.STOPWORDS)
    stopwords.add("said")

    wc_ref = wordcloud.WordCloud(
        background_color="white", max_words=2000, mask=mask_ref,
        stopwords=stopwords, contour_width=3, contour_color='steelblue'
    )
    wc_trs = wordcloud.WordCloud(
        background_color="white", max_words=2000, mask=mask_trs,
        stopwords=stopwords, contour_width=3, contour_color='steelblue'
    )

    # generate word cloud
    wc_ref.generate(' '.join(ref.to_list()))
    wc_trs.generate(' '.join(trans.to_list()))
    # store to file
    wc_ref_img = wc_ref.to_image()
    wc_trs_img = wc_trs.to_image()

    wc_img = Image.new('RGB', (w, h))
    wc_img.paste(wc_ref_img, (0, 0))
    wc_img.paste(wc_trs_img, (wc_ref_img.width, 0))
    if save:
        wc_img.save('image/' + fname)

    return wc_img


def visualize_metrics(metrics, metric_names, save=False, fname='image/metrics.png'):
    fig, ax = plt.subplots(2, 2, figsize=(2*4, 2*3))
    ax = ax.ravel()

    i = 0
    model_names = list(metrics.keys())
    n_models = len(model_names)

    for i, metric_name in enumerate(metric_names):
        cur_metric_values = []
        for model_name, model_metrics in metrics.items():
            cur_metric_values.append(model_metrics[metric_name])

        ax[i].set_ylabel("Values")
        ax[i].set_title(metric_name.replace('_', ' ').capitalize())
        ticks = range(n_models)
        ax[i].set_xticks(ticks, labels=model_names, rotation='horizontal')
    plt.tight_layout()
    if save:
        plt.savefig(fname)
    else:
        plt.show()
    plt.clf()