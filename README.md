# The Effect of Payment for Order Flow on Market Order Execution Quality

> *This paper examines the effect of payment for order flow on the routing of market orders for equities. I use variation in market center execution quality, obtained from SEC Rule 605 disclosures, to study differences in order routing between brokers who do and do not accept payment for order flow. My results suggest that payment for order flow causes a significant decrease in execution quality by decreasing responsiveness to changes in price improvement. Moreover, retail investors, on aggregate, lose more in missed potential price improvement than brokers gain in revenue from payment for order flow.*

The full [paper](documents/final.pdf) and a [presentation](documents/honors_presentation.pdf) on the results are in the documents folder. 

## Data

The data used for this paper comes from two sources: 605 disclosures and 606 disclosures; detailed information about these reports is available [here](https://www.sec.gov/divisions/marketreg/disclosure.htm). 

Aggregated raw data is located in the data folder as [rawdata_605.csv](data/rawdata_605.csv) and [rawdata_606.csv](data/rawdata_606.csv). This data is cleaned up to produce [605_processed.csv](data/processed/605_processed.csv) and [606_processed.csv](data/processed/606_processed.csv). These two sets of data are merged to produce the data for the regressions; [regression_data_levels.csv](data/processed/regression_data_levels.csv) is panel data, and [regression_data_levels_demeaned.csv](data/processed/regression_data_levels.csv) is the same data but demeaned within each exchange, broker, and market center subgroup. 

## Reproducing the Results

Each link redirects to the code (or source) used to produce its respective figure/table.

### Paper

* [Figure 1: Routing Venue Market Share over Time](data/code/visualizations/paper.R)
* [Table 1: Summary Statistics for Execution Quality Variables from 605 Data](notebooks/SummaryStats.ipynb)
* [Table 2: Example of 606 Disclosure (TD Ameritrade, NASDAQ, 2017Q3)](https://web.archive.org/web/20171204105513/https://www.tdameritrade.com/disclosure.page)
* [Table 3: Tobit Regression Results](analysis/code/parametric_tobit.do)
* [Table 4: SLS Regression Results](analysis/code/semiparametric_model.do)
* [Figures 2-6: Differences in Marginal Effects Between Brokers](analysis/code/semiparametric_me_plots.py)
* [Table A1: OLS Regression Results](analysis/code/parametric_ols.do)
* [Table A2: Broker Order Routing Averages](notebooks/SummaryStats.ipynb)
* [Table A3: Market Center Executions and Average Execution Quality](notebooks/SummaryStats.ipynb)
* [Table A4: Market Center Codes](notebooks/SummaryStats.ipynb)

### Presentation

The tables and figures used in the presentation are the same as those from the paper. 

* [Descriptive Statistics - Price Improvement](data/code/visualizations/presentation.R)
* [Descriptive Statistics - Execution Speed](data/code/visualizations/presentation.R)





