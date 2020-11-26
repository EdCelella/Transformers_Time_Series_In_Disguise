# Time Series Forecasting Using Transformers

This project applies an altered transformer architecture to the task of time-series forecasting. Specifically, the model was trained to forecast future stock prices of the S&P500 and DOW stock indexes. The program also includes the ability for different technical analysis techniques to be applied to the data, in order to produce different representations of the data. Namely, the two techniques implemented are moving averages, and chart pattern recognition.

A full description of the project can be found in the file [Transformers Time Series in Disguise](Transformers Time Series in Disguise.pdf), which is the research paper submitted alongside this codebase.

This project is in partial fulfilment for the degree of Master of Science in Advanced Computer Science, awarded by the University of Birmingham.

## Prerequisites

The project is written in Python 3. Thus, in order to run the code, Python 3 must be available on the machine as well as the following libraries:

-  [Jupyter Notebook](https://jupyter.org)
- [Scipy](https://www.scipy.org)
- [Pandas](https://pandas.pydata.org)
- [Matplotlib](https://matplotlib.org)
- [NumPy](https://numpy.org)
- [Pytorch](https://pytorch.org)

## How To Use

To run the project a Jupyter notebook server requires running in the Code directory, in order to launch the main.ipynb. From this file, all project code can be run. To run a specific test, the following steps must be taken:

1. Run the Imports cell.
2. Uncomment the filename for the data set you wish to use.
3. Choose a type of prepossessing to perform on the data from the regression
4. or classification data set subsections. **Only run one of these cells.**
5. Run all cells within the Prepare Set subheading.
6. For the Transformer, run the first cell in the Transformer section. Then run the cell commented either regression model, or classification model. The type of model you select must match the prepossessing type ran (e.g. if a cell in the Regression Dataset subheading was run, the regression model must be selected). Once a model has been selected run the last cell in the Transformer section. Once this has all be completed run the cell in the Transformer training subsection to run the selected test.
7. To run a test using the baseline RNN model, follow the same method described in step 5 of this guide within the Baseline model section. Note that due to this model using a pre-built RNN from the Pytorch library, training may need to be manually set. This varies depending on which device the training takes place.

## License

This project is licensed under the terms of the [Microsoft Reference Source License (MS-RSL)](License.md) and [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International](License.md).