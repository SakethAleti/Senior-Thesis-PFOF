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

** Regressions
xtset id

* Reg 1

xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd ///
	primp_avgt primp_avgt_rd, fe 

estimates store n1

xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd ///
	primp_avgt primp_avgt_rd, re
	
hausman n1 

if r(p) < 0.05 {

	xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd ///
		primp_avgt primp_avgt_rd, fe vce(robust)
		
	estimates store n1
	
} 
else {

	xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd ///
		primp_avgt primp_avgt_rd, re vce(robust)
		
	estimates store n1
	
}

* Reg 2

xtreg mktshare primp_expamt primp_expamt_rd primp_avgt primp_avgt_rd, fe 

estimates store n2

xtreg mktshare primp_expamt primp_expamt_rd primp_avgt primp_avgt_rd, re
	
hausman n2 

if r(p) < 0.05 {

	xtreg mktshare primp_expamt primp_expamt_rd primp_avgt primp_avgt_rd, fe vce(robust)
		
	estimates store n2
	
} 
else {

	xtreg mktshare primp_expamt primp_expamt_rd primp_avgt primp_avgt_rd, re vce(robust)
		
	estimates store n2
	
}

* Reg 3
	
xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd all_avgt ///
	all_avgt_rd, fe 

estimates store n3

xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd all_avgt ///
	all_avgt_rd, re
	
hausman n3 

if r(p) < 0.05 {

	xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd all_avgt ///
		all_avgt_rd, fe vce(robust)
		
	estimates store n3
	
} 
else {

	xtreg mktshare primp_pct primp_pct_rd primp_avgamt primp_avgamt_rd all_avgt ///
		all_avgt_rd, re vce(robust)
		
	estimates store n3
	
}

* Reg 4

xtreg mktshare primp_expamt primp_expamt_rd all_avgt all_avgt_rd, fe 

estimates store n4

xtreg mktshare primp_expamt primp_expamt_rd all_avgt all_avgt_rd, re
	
hausman n4 

if r(p) < 0.05 {

	xtreg mktshare primp_expamt primp_expamt_rd all_avgt all_avgt_rd, fe vce(robust)
		
	estimates store n4
	
} 
else {

	xtreg mktshare primp_expamt primp_expamt_rd all_avgt all_avgt_rd, re vce(robust)
		
	estimates store n4
	
}

* Results
esttab n1 n2 n3 n4, se r2 ar2 scalars(F)
