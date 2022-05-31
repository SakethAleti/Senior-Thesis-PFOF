library(ggplot2)
library(tidyverse)
library(ggthemes)
library(scales)
library(tikzDevice)
library(reshape)

# Market Share TAQ Data

filename <- '../../processed/overall_nms_mkt_shr_2017_03_22.csv'
rawdata <- read_csv(filename)

rawdata$date <- as.Date(as.character(rawdata$date), format = "%Y%m%d")

rawdata <- plyr::rename(rawdata, c(
  "D_vol" = "Off Exchange", 
  "nysegrp_vol" = "NYSE",
  "nasdaqgrp_vol" = "NASDAQ",
  "batsgrp_vol" = "BATS",
  "other_vol" = "Other"))

data <- melt(select(as.data.frame(rawdata), date, 
                    NYSE,
                    NASDAQ,
                    BATS,
                    'Off Exchange'), 
             id="date")


## aggregate by month


cbPalette <- c("#E69F00", "#56B4E9", "#009E73", "#494949", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

tikz(file  = '../../../exhibits/descriptive/nms_mkt_shr.tex',
     height = 3, width = 6.5)

ggplot(data = 
         filter(data[order(data$date),], 
                date > as.Date("2004-03-01")), 
       aes(x = date, y = value, color = variable)) + 
  geom_path(size= 1) + 
  #scale_y_continuous(labels = percent) +
  scale_x_date(breaks = date_breaks("2 years"), labels = date_format("%Y")) +
  xlab('Date') + ylab('Market Share') + 
  labs(#title = 'Market Share of Exchanges over Time',
    color = 'Routing Venue') +
  theme(panel.background = element_rect(fill = "white", 
                                        colour = NA), 
        panel.border = element_rect(fill = NA, colour = "grey20"), 
        panel.grid.major = element_line(colour = "grey87"), 
        panel.grid.minor = element_line(colour = "grey87", 
                                        size = 0.25), 
        strip.background = element_rect(fill = "grey85", 
                                        colour = "grey20"), 
        legend.key = element_rect(fill = "white", 
                                  colour = NA), 
        legend.position="bottom",
        legend.box.background = element_rect(size = 1),
        complete = TRUE,
        plot.margin = unit(c(0.2,1.8,0.5,0.2), "cm")) +
  guides(colour = guide_legend(title.position="top", title.hjust = 0.5))
#theme_bw(panel.grid.major = element_line(colour = "grey72")) 

# this takes a while (few min)
dev.off()
