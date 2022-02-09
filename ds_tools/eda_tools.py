from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



sns.set_style("whitegrid")
sns.set(palette="bright", font_scale=1.25)
plt.rcParams["figure.figsize"] = (15, 8)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ':'
plt.rcParams["grid.color"] = 'gray'



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
      bins (int or numpy.array): number of bins or numpy array of bins borders
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
        plt.plot(0, df[item].mean(), 'ow',
                 markersize=6, markeredgecolor='k')
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
            title_str = f"Spearman's corr.coeff. = {spearman_coeff:.3g}\n"
            title_str += f"Pearson's corr.coeff. = {pearson_coeff:.3g}"
            plt.title(title_str, fontsize=fontsize)

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
        fig, axes = plt.subplots(1, 1, figsize=(1+0.6*len(bar_data), 8))
        print((15, 1 * len(bar_data)))
        numerated_barplot(bar_data, axes, item,
                          color=color, fontsize=fontsize)

        ### ==== FIG 2 (boxplots for classes)
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        item_range = df[target_name].max() - df[target_name].min()
        y_min = df[target_name].min() - 0.05 * item_range
        y_max = df[target_name].max() + 0.05 * item_range
        sns.boxplot(x=item, y=target_name, data=df, ax=axes[0],
                    showmeans=True,
                    meanprops={'marker': 'o',
                               'markerfacecolor': 'white',
                               'markeredgecolor': 'black',
                               'markersize': 6})
        axes[0].set_xticklabels(bar_data.index, rotation=90)
        plt.grid(linestyle=':', color='gray')
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



def corr_analysis(df: pd.DataFrame,
                  target_name: str,
                  corr_type: str = 'spearman',
                  low_lim: float = 0.1,
                  high_lim: float = 0.8,
                  visualize_full_matrix: bool = False,
                  visualize_low_scatter: bool = True,
                  visualize_high_matrix: bool = False,
                  visualize_high_scatter: bool = True,
                  shrink_full: float = 0.82,
                  shrink_high: float = 0.82,
                  fontsize: int = 14) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    This function
    - calculates correlation matrix (Pearson's, or Spearman's, or something else)
    - plots heatmap of full correlation matrix (if `visualize_full_matrix=True`)
    - plots scatter plots for pairs of feature-target
        that have low correlation coefficients
        (absolute values less or equal `low_lim`)
        with linear approximations for these pairs
        (if `visualize_low_scatter=True`)
    - plots heatmap of correlation matrix for only features
        which have high-correlated coefficients
        (absolute values greater or equal `high_lim`)
        (if `visualize_high_matrix=True`)
    - plots scatter plots for pairs of features
        that have high correlation coefficients
        (absolute values greater or equal `high_lim`)
        with linear approximations for these pairs
        (if `visualize_high_scatter=True`)

    Requirements:
        matplotlib.pyplot as plt
        numpy as np
        pandas as pd
        seaborn as sns

    Args:
      df (pandas.DataFrame): dataset
      target_name (str): name of target column
      corr_type (str): type of correlation
          [‘pearson’, ‘kendall’, ‘spearman’ or callable - see docs for
          `pandas.DataFrame.corr` -
          https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html]
          (default = 'spearman')
      low_lim (float): the upper threshold for sorting out
          of low-correlated pairs of feature-target pairs
          (absolute values less or equal `low_lim`) (default = 0.1)
      high_lim (float): the low threshold for sorting out
          of high-correlated pairs of feature-target pairs
          (absolute values greater or equal `high_lim`) (default = 0.8)
      visualize_full_matrix (bool): flag defining whether plot
          heatmap of full correlation matrix or not (default = False)
      visualize_low_scatter (bool): flag defining whether plot
          scatter plots for pairs of feature-target
          that have low correlation coefficients
          (absolute values less or equal `low_lim`)
          with linear approximations for these pairs
          or not (default = True)
      visualize_high_matrix (bool): flag defining whether plot
          heatmap of correlation matrix for only features
          which have high-correlated coefficients
          (absolute values greater or equal `high_lim`)
          or not (default = False)
      visualize_high_scatter (bool): flag defining whether plot
          scatter plots for pairs of features
          that have high correlation coefficients
          (absolute values greater or equal `high_lim`)
          with linear approximations for these pairs
          or not (default = True)
      shrink_full (float): parameter for aligning the height of colorbar for
          heatmap of full correlation matrix with the height of the heatmap
          (default = 0.82)
      shrink_high (float): parameter for aligning the height of colorbar for
          heatmap of correlation matrix for only features
          which have high-correlated coefficients
          (absolute values greater or equal `high_lim`)
          with the height of the heatmap (default = 0.82)
      fontsize (int): font size used in figure titles (default = 14)

    Returns:
      [pd.DataFrame]: correlation matrix
      [pd.DataFrame]: Table with high-correlated pairs of features
        (feature names, correlation coefficients of this pair
        and each feature with target)
      [pd.Series]: feature-target correlation coefficients
        for low-correlated features
    """

    assert 0 < high_lim < 1, \
        "`high_lim` should follows by `0 < high_lim < 1`"
    assert 0 <= low_lim < 1, \
        "`low_lim` should follows by `0 <= low_lim < 1`"


    ### === Full matrix
    corr_matrix = df.corr(corr_type)

    if visualize_full_matrix:
        plt.subplots(1, 1, figsize=(16, 16))
        ## Generate a custom diverging colormap
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)
        ## Generate a blind mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, center=0,
                    cmap=cmap, cbar=True,
                    cbar_kws={"shrink": shrink_full,
                              "ticks": list(np.round(np.arange(-1, 1.1, 0.2), 1))},
                    vmin=-1, vmax=1,
                    square=True, linewidths=0.1, linecolor='k',
                    annot=True, annot_kws={'fontsize': 'small'}, fmt='.2f')
        plt.xticks(rotation=90)
        plt.title(corr_type.title() + "'s' correlation matrix: $C_{ij}$",
                  fontsize=fontsize)
        plt.show()


    ### === Matrix of low correlations with target
    mask_target_low = abs(corr_matrix[target_name]) <= low_lim
    corr_matrix_target_low = corr_matrix[mask_target_low][target_name]
    corr_matrix_target_low = corr_matrix_target_low.sort_values()

    if len(corr_matrix_target_low) > 0:
        print_str = ">>> These features have low correlation coefficients with target:"
        print_str += f"(|C| <= {low_lim})."
        print(print_str)
        if visualize_low_scatter:
            for feat in corr_matrix_target_low.index:
                df_low = df[[feat, target_name]].dropna()

                range_x = df_low[feat].max() - df_low[feat].min()
                x_min = df_low[feat].min() - 0.03 * range_x
                x_max = df_low[feat].max() + 0.03 * range_x
                range_y = df_low[target_name].max() - df_low[target_name].min()
                y_min = df_low[target_name].min() - 0.03 * range_y
                y_max = df_low[target_name].max() + 0.03 * range_y

                plt.subplots(1, 2, figsize=(12, 6))

                plt.subplot(1, 2, 1)
                sns.regplot(x=feat, y=target_name, data=df_low,
                            scatter_kws={'s': 9}, color='forestgreen')
                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))
                plt.grid(linestyle=':', color='gray')
                plt.xlabel(feat, fontsize='small')
                plt.ylabel(target_name, fontsize='small')
                title_str = f"Dependence of `target` on `{feat}`:\n"
                title_str += "$C_{i, target} =$ "
                title_str += f"{corr_matrix_target_low.loc[feat]:.3g}"
                plt.title(title_str, fontsize='small')

                plt.subplot(1, 2, 2)
                plt.hexbin(df_low[feat], df_low[target_name],
                           gridsize=50, vmin=0, cmap="Greens", alpha=0.6)
                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))
                plt.grid(linestyle=':', color='gray')
                plt.xlabel(feat, fontsize='small')
                title_str = f"Data distribution in `target` - `{feat}` plane"
                plt.title(title_str, fontsize='small')
                plt.show()
        else:
            display(corr_matrix_target_low)
    else:
        print(f""">>> There are not feature with correlation coefficients: 
              |C| <= {low_lim}.""")


    ### === "High-value" corr matrix
    corr_matrix_high = corr_matrix[abs(corr_matrix) >= high_lim]
    ## drop empty rows and columns as well as digonal elements
    for col in corr_matrix_high.columns:
        corr_matrix_high.loc[col, col] = np.nan
    corr_matrix_high = corr_matrix_high.dropna(axis=0, how='all')
    corr_matrix_high = corr_matrix_high.dropna(axis=1, how='all')

    ### --- Table of high-correlated pairs
    high_corr_df = []
    for ind in corr_matrix_high.index:
        for col in corr_matrix_high.columns:
            if abs(corr_matrix_high.loc[ind, col]) > 0:
                high_corr_df.append([ind,
                                     col,
                                     str(sorted(list(set([ind, col])))),
                                     corr_matrix_high.loc[ind, col],
                                     corr_matrix.loc[ind, target_name],
                                     corr_matrix.loc[col, target_name]])
    high_corr_df = pd.DataFrame(high_corr_df,
                                columns=['var1', 'var2', 'var_set',
                                         'corr_coeff_1_2',
                                         'corr_coeff_1_target',
                                         'corr_coeff_2_target'])
    high_corr_df = high_corr_df.drop_duplicates(subset='var_set')
    high_corr_df = high_corr_df.drop('var_set', axis=1)
    high_corr_df = high_corr_df.sort_values('corr_coeff_1_2', ascending=False)
    high_corr_df = high_corr_df.reset_index(drop=True)

    if len(high_corr_df) > 0:
        print_str = ">>> These pairs of features have high correlation coefficients:"
        print_str += f"(|C| >= {high_lim})."
        print(print_str)

        ### === Draw heatmaps with the masks and "square" aspect ratio
        if visualize_high_matrix:
            plt.subplots(1, 1, figsize=(len(corr_matrix_high),
                                        len(corr_matrix_high)))
            ## Generate a blind mask for the upper triangle
            mask = np.triu(np.ones_like(corr_matrix_high, dtype=bool))
            sns.heatmap(corr_matrix_high,
                        mask=mask,
                        center=0,
                        cmap=sns.color_palette("coolwarm", as_cmap=True),
                        cbar=True,
                        cbar_kws={"shrink": shrink_high,
                                  "ticks": list(np.round(np.arange(-1, 1.05, 0.2), 1))},
                        vmin=-1, vmax=1,
                        annot=True, annot_kws={'fontsize': 'small'}, fmt='.2g',
                        square=True, linewidths=0.1, linecolor='k')
            plt.xticks(rotation=90)
            title_str = f"{corr_type.title()}'s correlation matrix:"
            title_str += "$C_{ij}: |C_{ij}|\geq$" + f"{high_lim}"
            plt.title(title_str, fontsize=fontsize)
            plt.show()

        if visualize_high_scatter:
            for ind in high_corr_df.index:
                var1 = high_corr_df.loc[ind, 'var1']
                var2 = high_corr_df.loc[ind, 'var2']
                df_high = df[[var1, var2]].dropna()
                range_x = df_high[var1].max() - df_high[var1].min()
                x_min = df_high[var1].min() - 0.03 * range_x
                x_max = df_high[var1].max() + 0.03 * range_x
                range_y = df_high[var2].max() - df_high[var2].min()
                y_min = df_high[var2].min() - 0.03 * range_y
                y_max = df_high[var2].max() + 0.03 * range_y

                plt.subplots(1, 2, figsize=(12, 6))
                plt.subplot(1, 2, 1)
                sns.regplot(x=var1, y=var2, data=df_high,
                            scatter_kws={'s': 9}, color='maroon',
                            label="Linear approximation")
                plt.legend(fontsize='x-small')
                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))
                plt.xlabel(var1, fontsize='small')
                plt.ylabel(var2, fontsize='small')
                title_str = f"Dependence of `{var2}` on `{var1}`:\n"
                title_str += "$C_{ij} = $"
                title_str += f"{high_corr_df.loc[ind, 'corr_coeff_1_2']:.3g}"
                plt.title(title_str, fontsize='small')

                plt.subplot(1, 2, 2)
                plt.hexbin(df_high[var1], df_high[var2], gridsize=50, vmin=0,
                           cmap="Reds", alpha=0.6)
                plt.xlim((x_min, x_max))
                plt.ylim((y_min, y_max))
                plt.xlabel(var1, fontsize='small')
                plt.xlabel(var1, fontsize='small')
                title_str = f"Data distribution in `{var1}` - `{var2}` plane\n"
                if target_name not in [var1, var2]:
                    title_str += f"{var1}: " + "$C_{i, target} = $"
                    title_str += f"{corr_matrix.loc[var1, target_name]:.3f}\n"
                    title_str += f"{var2}: " + "$C_{i, target} = $"
                    title_str += f"{corr_matrix.loc[var2, target_name]:.3f}"
                plt.title(title_str, fontsize='small')
                plt.show()
        elif not (visualize_high_matrix or visualize_high_scatter):
            display(high_corr_df)

    else:
        print_str = ">>> There are not pairs of features with high correlation coefficients:" \
                  + f"(|C| >= {high_lim})."
        print(print_str)

    return corr_matrix, high_corr_df, corr_matrix_target_low
