import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def num_variable_analysis(df, item, target_name, target_type,
                          bins=100, color='maroon', fontsize=14,
                          descr_dict=None, data_info=None):
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
      bins (int or numpy.array): number of bins or numpy arrary of bins borders
          for histplot (default=100)
      color (str or sequence of numbers): color of plots (defult='maroon')
          [see matplotlib docs for details]
      fontsize (int): font size used in figure titles and legends (fontsize-1)
          (default=14)
      descr_dict (dict): [optional] dictionary where keys are column names
          (including `item`) and values contain some conclusions
          about these features - to be printed (default=None).
      data_info (pandas.DataFrame): [optional] table indexing with column names
          (including `item`), where columns contain additional
          info about `item` - to be printed (default=None).

    Returns:
    - if target type is 'numeric' and item != target_name
        - Pearson's and Spearman's correlation coefficients
            for feature/target pair [tuple of floats]
    - else
        - None
    """

    assert target_type in ['numeric', 'categorical'], \
        "Please define `target_type` as 'numeric' or 'categorical'"

    if item != target_name:
        nx = 3
    else:
        nx = 2

    result = None

    fig, axes = plt.subplots(1, nx, figsize=(15, 8), sharey=True)
    item_range = df[item].max() - df[item].min()
    y_min = df[item].min() - 0.05 * item_range
    y_max = df[item].max() + 0.05 * item_range

    ### ==== FIG 1 (histplot)
    plt.subplot(1, nx, 1)
    sns.histplot(data=df, y=item, bins=bins, kde=True, color=color)
    plt.ylim((y_min, y_max))
    plt.xticks(rotation=90)
    plt.title(f"Distribition of {item}", fontsize=fontsize)

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
                plt.legend(fontsize=fontsize - 1)
        plt.ylim(y_min, y_max)

    ### === Descriptive statistics
    describer = pd.DataFrame(df[item].describe()).T
    print(f"==== {item} ====")
    try:
        display(pd.DataFrame(data_info.loc[item, :]))
    except:
        print(f"There are {df[item].isna().sum()} missing values in '{item}'\n.")

    print(">>> Statistics:")
    display(describer)

    ### === Correlation coefficients for feature/target
    if item != target_name:
        if target_type == 'numeric':
            pearson_coeff = df[[item, target_name]].corr().loc[item, target_name]
            spearman_coeff = df[[item, target_name]].corr(method='spearman').loc[item, target_name]
            print(">>> Correlation:")
            print(f"  Pearson's  correlation coefficient between '{item}' and '{target_name}' is {pearson_coeff:.3g}.")
            print(f"  Spearman's correlation coefficient between '{item}' and '{target_name}' is {spearman_coeff:.3g}.\n\n")
            result = (pearson_coeff, spearman_coeff)
            plt.title(f"Spearman's corr.coeff. = {spearman_coeff:.3g}\n" \
                      + f"Pearson's corr.coeff. = {pearson_coeff:.3g}",
                      fontsize=fontsize)

    plt.show()

    try:
        print(">>> CONCLUSION:", descr_dict[item])
    except:
        pass
    print("\n" * 2)

    return result
