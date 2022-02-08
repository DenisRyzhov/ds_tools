from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



def num_variable_analysis(df: pd.DataFrame,
                          item: str,
                          target_name: str,
                          target_type: str,
                          bins: int or Iterable[float] = 100,
                          color: str = 'forestgreen',
                          fontsize: int = 14,
                          descr_dict: Optional[Dict[str, str]] = None,
                          data_info: Optional[pd.DataFrame] = None) \
        -> Optional[Tuple[float]]:
    """
    This function
    - plots distribution with `sns.histplot()`
    - plots `sns.boxenplot()` for each class (distribution of target),
        it helps to detect anomalies inside each class
    - calculates the basic statistical indicators `.describe ()`
    - calculates the number of missing values

    - if target type is 'numeric'
      - plots scatter plot for feature/target pair to find out dependencies
      - calculates Pearson's and Spearman's correlation coefficients
          for feature/target pair

    - if target type is 'categorical'
        - plots kdeplot for values of categorical target
            to detect differences in distribution

    Requirements:
      matplotlib.pyplot as plt
      pandas as pd
      seaborn as sns
      scipy.stats

    Args:
      df (pandas.DataFrame): dataset
      item (str): name of column to investigate
      target_name (str) :name of target feature
      target_type (str): type of target variable: 'numeric' or 'categorical'
      bins (int or numpy.array): number of bins or numpy arrray of bins borders
          for sns.histplot (default = 100)
      color (str or sequence of numbers): color of plots (default = 'forestgreen')
          [see matplotlib docs for details]
      fontsize (int): font size used in figure titles (default = 14)
      descr_dict (dict): [optional] dictionary where keys are column names
          (including `item`) and values contain some conclusions
          about these features - to be printed (default = None).
      data_info (pandas.DataFrame): [optional] table indexing with column names
          (including `item`), where columns contain additional
          info about `item` - to be printed (default = None).

    Returns:
    - if target type is 'numeric' and item != target_name
        - [tuple of floats] Pearson's and Spearman's correlation coefficients
            for feature/target pair
    - else
        - None
    """

    ### === Descriptive statistics
    describer = pd.DataFrame(df[item].describe()).T
    print(f"==== {item} ====")
    try:
        display(pd.DataFrame(data_info.loc[item, :]))
    except:
        print(f"There are {df[item].isna().sum()} missing values in '{item}'\n.")

    print(">>> Statistics:")
    display(describer)

    assert target_type in ['numeric', 'categorical'], \
        "Please define `target_type` as 'numeric' or 'categorical'"

    if item != target_name:
        nx = 3
    else:
        nx = 2

    result = None

    plt.subplots(1, nx, figsize=(15, 8), sharey=True)
    # fig, axes = plt.subplots(1, nx, figsize=(15, 8), sharey=True)
    item_range = df[item].max() - df[item].min()
    y_min = df[item].min() - 0.05 * item_range
    y_max = df[item].max() + 0.05 * item_range

    ### ==== FIG 1 (histplot)
    plt.subplot(1, nx, 1)
    sns.histplot(data=df, y=item, bins=bins, kde=True, color=color)
    plt.ylim((y_min, y_max))
    plt.xticks(rotation=90)
    plt.title(f"Distribution of {item}", fontsize=fontsize)

    ### ==== FIG 2 (boxenplot)
    plt.subplot(1, nx, 2)
    if target_type == 'numeric':
        sns.boxenplot(y=df[item], orient='v', color=color)
    else:
        sns.boxenplot(x=df[target_name], y=df[item], orient='v')
        plt.xticks(rotation=90)
    plt.ylim((y_min, y_max))
    plt.ylabel("")
    plt.title(item, fontsize=fontsize)

    ### === FIG 3 (scatterplot for numeric target
    ###            OR kdeplot for values of categorical target)
    if item != target_name:
        plt.subplot(1, nx, 3)
        if target_type == 'numeric':
            plt.plot(df[target_name], df[item], 'o',
                     markersize=3, markeredgecolor=color, markerfacecolor=color)
            plt.xticks(rotation=90)
        elif target_type == 'categorical':
            values_targ = df[target_name].unique()
            for value_targ in values_targ:
                sns.kdeplot(df[df[target_name] == value_targ][item],
                            vertical=True,
                            label=f"{value_targ}: {len(df[df[target_name] == value_targ])}")
                plt.title(item, fontsize=fontsize)
                plt.legend(fontsize='small')
        plt.ylim(y_min, y_max)

    ### === Correlation coefficients for feature/target
    if item != target_name:
        if target_type == 'numeric':
            pearson_coeff = df[[item, target_name]].corr().loc[item, target_name]
            spearman_coeff = df[[item, target_name]].corr(method='spearman').loc[item, target_name]
            print(">>> Correlation:")
            print(f"  Pearson's  correlation coefficient between '{item}' and '{target_name}' is {pearson_coeff:.3g}.")
            print(f"  Spearman's correlation coefficient between '{item}' and '{target_name}' is {spearman_coeff:.3g}.")
            result = (pearson_coeff, spearman_coeff)
            plt.title(f"Spearman's corr.coeff. = {spearman_coeff:.3g}\n" \
                      + f"Pearson's corr.coeff. = {pearson_coeff:.3g}",
                      fontsize=fontsize)

    plt.show()

    try:
        print(">>> CONCLUSION:", descr_dict[item], "\n" * 2)
    except:
        print("\n" * 2)

    return result



def categ_variable_analysis(df: pd.DataFrame,
                            item: str,
                            item_type: str,
                            target_name: str,
                            target_type: str,
                            color: str = 'forestgreen',
                            fontsize: int = 14,
                            descr_dict: Optional[Dict[str, str]] = None,
                            data_info: Optional[pd.DataFrame] = None) \
        -> [pd.Series or Tuple[pd.Series, pd.DataFrame]]:
    """
    This function
    - plots distribution along classes in fact with `sns.barplot()`
        - helps to detect class imbalance
    - calculates the basic statistical indicators `.describe ()`
    - calculates the number of missing values
    - calculates populations for classes

    - if target type is 'categorical' (or 'one_hot', or 'ordinal')
      - plots `sns.countplot()` for each class (distribution of target),
          it helps to detect anomalies inside each class
      - calculates contingency table for feature/target pair

    - if target type is 'numeric'
      - plots boxplots and KDE plots for each class

    Requirements:
        matplotlib.pyplot as plt
        pandas as pd
        seaborn as sns

    Args:
      df (pandas.DataFrame): dataset
      item (str): name of column to investigate
      item_type (str): type of target variable: 'one_hot', 'ordinal'
      target_name (str): name of target column
      target_type (str): type of target variable: 'numeric' or 'categorical'
      color (str or sequence of numbers): color of barplots (default = 'forestgreen')
          [see matplotlib docs for details]
      fontsize (int): font size used in figure titles (default = 14)
      descr_dict (dict): [optional] dictionary where keys are column names
          (including `item`) and values contain some conclusions
          about these features - to be printed (default = None).
      data_info (pandas.DataFrame): [optional] table indexing with column names
          (including `item`), where columns contain additional
          info about `item` - to be printed (default = None).

    Returns:
      [pd.Series]: population for classes
    - if item != target_name and target type is one of
        ['categorical', 'one_hot', 'ordinal']
          [pd.Series]: population for classes
          [pd.DataFrame]: contingency table for feature/target pair
    """

    assert item_type in ['one_hot', 'ordinal'], \
        "`item_type` should be 'one_hot' or 'ordinal'."
    assert target_type in ['categorical', 'one_hot', 'ordinal', 'numeric'], \
        "`target_type` should be one of ['categorical', 'one_hot', 'ordinal', 'numeric']."

    if item_type == 'ordinal':
        bar_data = df[item].value_counts().sort_index()
    else:
        bar_data = df[item].value_counts().sort_values(ascending=False)

    result = bar_data

    ### === Descriptive statistics
    describer = pd.DataFrame(df[item].describe()).T
    print(f"==== {item} ====")
    try:
        display(pd.DataFrame(data_info.loc[item, :]))
    except:
        print(f"There are {df[item].isna().sum()} missing values in '{item}'\n.")
    print(">>> Statistics:")
    display(describer)
    print(">>> Category counts:")
    print(bar_data)

    if item == target_name:
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))

        ### ==== FIG 1 (barplot)
        numerated_barplot(bar_data, axes, item,
                          color=color, fontsize=fontsize)

    elif target_type in ['categorical', 'one_hot', 'ordinal']:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        ### ==== FIG 1 (barplot)
        plt.subplot(1, 2, 1)
        numerated_barplot(bar_data, axes[0], item,
                          color=color, fontsize=fontsize)

        ### ==== FIG 2 (countplot for classes)
        sns.countplot(x=item, hue=target_name, data=df, ax=axes[1])
        plt.legend(loc='best', fontsize='small')
        axes[1].set_xticklabels(bar_data.index, rotation=90)

        ### contingency table
        contingency_table = pd.crosstab(df[target_name], df[item],
                                        normalize=True)
        plt.subplots(1, 1, figsize=(1.5 * contingency_table.shape[1],
                                    contingency_table.shape[0]))
        sns.heatmap(contingency_table, center=0, cmap='PiYG', cbar=True, vmin=0,
                    square=False, linewidths=0.1, linecolor='k',
                    annot=True, fmt='.3g')
        plt.title("Contingency table (frequencies)")

        result = bar_data, contingency_table

    elif target_type == 'numeric':

        ### ==== FIG 1 (barplot)
        fig, axes = plt.subplots(1, 1, figsize=(0.6 * len(bar_data), 8))
        print((15, 1 * len(bar_data)))
        numerated_barplot(bar_data, axes, item,
                          color=color, fontsize=fontsize)

        ### ==== FIG 2 (boxplots for classes)
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        item_range = df[target_name].max() - df[target_name].min()
        y_min = df[target_name].min() - 0.05 * item_range
        y_max = df[target_name].max() + 0.05 * item_range
        sns.boxplot(x=item, y=target_name, data=df, ax=axes[0])
        axes[0].set_xticklabels(bar_data.index, rotation=90)
        axes[0].set_ylim((y_min, y_max))
        axes[0].set_title(item, fontsize=fontsize)

        ### ==== FIG 3 (kdeplots for classes)
        for value in bar_data.index:
            if bar_data[value] > 1:
                sns.kdeplot(y=df[df[item] == value][target_name],
                            ax=axes[1],
                            label=f"{value} [{bar_data[value]} values]")
            else:
                plt.plot(y_min, 0,
                         label=f"{value} [{bar_data[value]} value - not plotted]")
        axes[1].legend(fontsize='small')
        axes[1].set_ylim((y_min, y_max))
        axes[1].set_ylabel("")
        axes[1].set_title(item, fontsize=fontsize)

    plt.show()

    try:
        print(">>> CONCLUSION:", descr_dict[item])
    except:
        pass
    print("\n" * 2)

    return result



def numerated_barplot(bar_data: pd.Series,
                      axis: plt.Axes,
                      item: str,
                      color: str = 'forestgreen',
                      fontsize: int = 14) -> None:
    """
    This function plots barplot with numbers indicated in plot for each bar.

    Requirements:
        matplotlib.pyplot as plt
        pandas as pd
        seaborn as sns

    Args:
      bar_data (pd.Series): pandas.Series with names of classes as indices
          and populations of classes as values
      axis (plt.Axes): matplotlib axis
      item: str,
      color (str or sequence of numbers): color of barplot contours
          (default = 'forestgreen') [see matplotlib docs for details]
      fontsize (int): font size used in figure title (default = 14)

    Returns:
      None
    """

    sns.barplot(x=bar_data.index, y=bar_data, ax=axis,
                linewidth=1.5, facecolor='w', edgecolor=color)
    for x, y in enumerate(bar_data):
        if y < bar_data.max() / 2:
            y_pos = y + 0.02 * bar_data.max()
        else:
            y_pos = y - 0.08 * bar_data.max()
        axis.text(x, y_pos, str(y),
                  horizontalalignment='center', fontsize='small', rotation=90)
    axis.set_xticklabels(bar_data.index, rotation=90)
    axis.set_title(item, fontsize=fontsize)
