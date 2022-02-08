from typing import Any, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns



def pdf_fit(x: np.ndarray or pd.Series,
            distrib: Type[Any] = scipy.stats.norm,
            visualize: bool = True,
            bins: int = 100,
            color_distr: str ='forestgreen',
            color_fit: str ='maroon',
            fontsize: int = 14,
            show_kde: bool = True,
            show_semilogy: bool = False,
            name: Optional[str] = None) -> Tuple[float]:
    """
    This function fits data
       with the certain statistical continuous distribution.

    Requirements:
        matplotlib.pyplot as plt
        pandas as pd
        seaborn as sns
        scipy
        seaborn as sns

    Args:
      x (numpy.array or pandas.Series): data
      distrib (function): name of the assumed distribution
          of the class `scipy.stats.rv_continuous` that has .fit() method.
          The list of possible distributions can be found here:
          https://docs.scipy.org/doc/scipy/tutorial/stats/continuous.html .
          (default = scipy.stats.expon)
      visualize (bool): flag of plot or not data distribution and its fitting
          (default = True)
      bins (int or numpy.array): number of bins or numpy arrray of bins borders
          for sns.histplot (default = 100)
      color_distr (str or sequence of numbers): color of distribution plots
          [see matplotlib docs for details] (default = 'forestgreen')
      color_fit (str or sequence of numbers): color of fitting PDF
          [see matplotlib docs for details] (default = 'maroon')
      show_kde (bool): flag of show or not KDE in sns.histplot
          (default = True)
      show_semilogy (bool): flag of plot or not data distribution and its fitting
          with semilogy axis (default = False)
      name (str): [optional] name of x for figure titles
          (if data are pandas.Series and name=None,
          the name of Series will be printed).

    Returns:
      tuple of fitted parameters (for details see
         https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit)
    """

    params = distrib.fit(x)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = distrib.pdf(x_fit, *params)

    if visualize:
        if name is None:
            try:
                name = f"'{x.name}'"
            except:
                name = ""

        ny = 1
        if show_semilogy:
            ny = 2

        plt.subplots(ny, 1, figsize=(15, ny * 7))

        plt.subplot(ny, 1, 1)
        sns.histplot(x, bins=bins, kde=show_kde, stat='density',
                     color=color_distr, label="data")
        plt.plot(x_fit, y_fit, '-', label="fitting", color=color_fit)
        plt.legend()
        plt.title(f"{name} distribution", fontsize=fontsize)

        if show_semilogy:
            plt.subplot(ny, 1, 2)
            sns.histplot(x, bins=bins, kde=show_kde, stat='density',
                         color=color_distr, label="data")
            plt.semilogy(x_fit, y_fit, '--',
                         label="fitting", color=color_fit)
            plt.legend()
            plt.title(f"{name} distribution (semilogy)", fontsize=fontsize)

        plt.show()

    return params