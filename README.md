# Current fluctuations in open quantum systems: Bridging the gap between quantum continuous measurements and full counting statistics

This repository contains all the code required to recreate all the figures in our manuscript, Current fluctuations in open quantum systems. 
The tutorial aims to provide a unified toolbox for describing current fluctuations in continuously measured quantum systems. The tools and techniques used to describe these fluctuations are scattered across different communities, and this tutorial brings them together to provide novel insights and practical analytical and numerical tools. You can find the most recent version of the manuscript on the [arXiv](https://arxiv.org/abs/2303.04270).

There are several python and Mathematica notebooks that accompany the tutorial. All the examples in these notebooks are organized according to the tutorial's convention, that is A, B, C, and D. The physical and mathematical details of each example can be found in Sec. 2B. These notebooks are designed to be used in conjunction with the manuscript which you should refer to for all mathematical and physical details. The key mathematical techniques required to numerical implement the results can be found throughout the manuscript, however there is a strong emphasis on Sec. 4 Full Counting Statistics, and Sec. 5 Solution Methods.

### Python 
There are 4 Jupyter notebooks titled Example_A, Example_B, Example_C, and Example_D which can be used to recreate many of the figures pertaining to each example. The notebook Butterworth_filters can be used to understand the signal processing techniques that are used in the manuscript.

The following packages are required to run the code in this tutorial:

- `numpy`
- `matplotlib`
- `scipy`
- `qutip`

To install these packages, you can use pip:

`pip install numpy matplotlib scipy qutip`

The primary functions are defined in FCS.py which can be imported into other notebooks.

### Mathematica
Here we provide the Mathematica code for the tutorial. The primary functions are defined in melt.nb and are demonstrated in Examples (public).nb and are organized according to the tutorial examples. 

You can also find many more additional examples discussed in the Examples (public) notebook.

You can find further details about the melt package at the following [website](https://melt1.notion.site/melt1/Melt-2d05fca5cfa342e28cafaf6fe26e049e). Melt is a self-contained package and no further downloads are required.





