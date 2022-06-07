import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn
seaborn.set()  # make the plots look pretty

from patent_phrase_similarity.data.transformation.cpc_datasets import Datasets, CPCDatasets

sns.set(style="darkgrid")
sns.set(font_scale=1.3)


def plot_categories(_df, column='section', figsize=None, top=0, ascending=True):
    df_group_section = cpc_train_df.groupby([column])
    section_freq = dict(df_group_section.section.count())
    _section_freq_df = pd.DataFrame(section_freq.items(), columns=[column, 'freq']).sort_values('freq', ascending=ascending)
    if top > 0:
        _section_freq_df = _section_freq_df.iloc[:top]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=column, y="freq", data=_section_freq_df, palette="PuBuGn_d", ax=ax)
    for i in ax.containers:
        ax.bar_label(i,)
    plt.show()


def plot_distribution(_df, column='score', facet_column='score', height=5, aspect=4):
    sns.FacetGrid(cpc_train_df, hue=facet_column, height=height, aspect=aspect).map(sns.kdeplot, column).add_legend()


def plot_correlation(_df, column='score', figsize=None):
    corr = cpc_train_df[column].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, ax=ax)
    plt.show()


def plot_correlation_matrix(_df, column='score', figsize=None):
    corr = _df[column].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, ax=ax)
    plt.show()


def plot_score_vs_test_score(_df, column_a='score', column_b='test_score', figsize=None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(x=column_a, y=column_b, data=_df, fit_reg=False, ax=ax)
    plt.show()


if __name__ == '__main__':
    datasets = Datasets()
    cpc_datasets = CPCDatasets()

    train_df = datasets.get_train_df()
    cpc_train_df = cpc_datasets.merge_with_df(train_df)
    #
    # sns.FacetGrid(cpc_train_df, hue="score", height=5, aspect=4).map(sns.kdeplot, "context_cat").add_legend()
    # sns.FacetGrid(cpc_train_df, hue="score", height=5, aspect=4).map(sns.kdeplot, "section_cat").add_legend()

    plot_categories(cpc_train_df, column='section', ascending=False)
    plot_categories(cpc_train_df, column='context', figsize=(40, 10), ascending=False, top=20)
