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


def plot_distance(
    df: pd.DataFrame,
    save=False,
    fname_diff='image/raw_histogram.png'
):
    plt.xlabel("Value")
    plt.ylabel("no. of entries")
    plt.title("Toxicity difference")
    plt.hist(df['ref_tox']-df["trn_tox"], 20)
    if save:
        plt.savefig(fname_diff)
    else:
        plt.show()
    plt.clf()


def plot_length_distributions(
    df: pd.DataFrame,
    save=False,
    fname_diff='image/length_distribution.png'
):
    plt.xlabel("Length")
    plt.ylabel("no. of entries")
    plt.title("Toxicity difference")
    for corpus in ['reference', 'translation']:
        length = np.array([len(s) for s in df[corpus]])
        elements, repears = np.unique(length, return_counts=True)
        plt.plot(elements[:200], repears[:200], label=corpus)
    plt.legend()
    if save:
        plt.savefig(fname_diff)
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