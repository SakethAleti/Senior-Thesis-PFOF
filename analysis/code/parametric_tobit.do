clear

* Import data
import delimited D:\Users\saketh\Documents\GitHub\Order-Routing-PFOF\data\processed\regression_data_levels.csv

* Update data
drop if ordertype != "Market"
egen id = group(broker marketcenter exchange)
gen qd = quarterly(quarter, "YQ")

gen all_avgt_rd = all_avgt * rebate_dummy
gen primp_avgt_rd = primp_avgt * rebate_dummy
gen primp_expamt_rd = primp_expamt * rebate_dummy
gen primp_avgamt_rd = primp_avgamt * rebate_dummy
gen primp_pct_rd = primp_pct * rebate_dummy

* Regressions
xtset id

xttobit mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd primp_avgt ///
	primp_avgt_rd, ll(0) 

estimates store n0
	
xttobit mktshare primp_expamt primp_expamt_rd primp_avgt primp_avgt_rd, ll(0) 

estimates store n1

xttobit mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd all_avgt ///
	all_avgt_rd, ll(0) 

estimates store n2
	
xttobit mktshare primp_expamt primp_expamt_rd all_avgt all_avgt_rd, ll(0)

estimates store n3

* Results
esttab n0 n1 n2 n3, se
