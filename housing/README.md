# Estimating Home Prices


## Notebook

To see my analysis checkout the ipython notebook -- github should be able to render it. 

You will see that we achieved a respectable R^2 score of 0.79 (+/- 0.04) by 10-fold CV evaluation metric. Also the mean error was measured at: 17.428%. We hope to achieve something close to this on the reserve dataset.

## Reserve dataset

The model you can run the reserve dataset against is called `model.py`. There is an example runner in `runner.py`. Put your reserve data into `test.csv` in the same exact format the test set you provide me. I left the headers in `test.csv` so you can make sure they align. Then just run `runner.py` and it will report back the reserve dataset evaluation metrics.

