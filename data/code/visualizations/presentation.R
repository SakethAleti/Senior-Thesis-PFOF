library(ggplot2)
library(tidyverse)
library(reshape2)
library(stargazer)
library(sandwich)
library(AER)
library(ggthemes)

## Params

filename_output <- "../../analysis/html_tables/tables_all_market.html"
folder <- '../../processed/'
table_name <- 'All Brokers'

data_df <- read_csv(paste(folder, 'regression_data_levels.csv', sep = ''))
data_df    <- filter(data_df,   OrderType == "Market")

data_605 <- read_csv(paste(folder, '../rawdata_605.csv', sep = ''))

## ------------------------------------------

data_605$PrImp_ExpAmt = data_605$PrImp_AvgAmt * data_605$PrImp_Pct

# price improv

ggplot(data = filter(data_605, OrderType == "Market" & idate >= 2017), 
       aes(x = PrImp_ExpAmt, fill = Exchange)) + 
  #geom_histogram(bins = 60) + 
  geom_density(alpha = 0.7) +
  xlab('Price Improvement ($/share)') + ylab('Density')  +
  scale_fill_ptol("Exchange") +
  theme_minimal() + 
  geom_vline(xintercept = 0, color = "gray35", size = 0.5) +
  geom_hline(yintercept = 0, color = "gray35", size = 0.5) +
  labs(title = 'Distribution of Expected Price Improvement for Market Orders in 2017',
       subtitle = 'Orders between 100-499 shares',
         fill = 'Exchange') 

ggsave("../../../exhibits/descriptive/primp_expamt.pdf", 
       width = 8, height = 4)

# execution speed

ggplot(data = filter(data_605, OrderType == "Market" & idate >= 2017 & PrImp_AvgT < 1), 
       aes(x = PrImp_AvgT, fill = Exchange)) + 
  #geom_histogram(bins = 60) + 
  geom_density(alpha = 0.7) +
  xlab('Execution Speed (s)') + ylab('Density')  +
  scale_fill_ptol("Exchange") +
  theme_minimal() + 
  geom_vline(xintercept = 0, color = "gray35", size = 0.5) +
  geom_hline(yintercept = 0, color = "gray35", size = 0.5) +
  labs(title = 'Distribution of Average Execution Time for Market Orders in 2017',
       subtitle = 'Orders between 100-499 shares',
       fill = 'Exchange') 

ggsave("../../../exhibits/descriptive/primp_avgt.pdf", 
       width = 8, height = 4)

