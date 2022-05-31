// Non-POF

sls mktshare primp_pct primp_avgamt primp_avgt ///
	if rebate_dummy == 0 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))
	
estimates store n1

sls mktshare primp_expamt primp_avgt ///
	if rebate_dummy == 0 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))
	
estimates store n2
	
sls mktshare primp_pct primp_avgamt all_avgt ///
	if rebate_dummy == 0 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))
	
estimates store n3

sls mktshare primp_expamt all_avgt ///
	if rebate_dummy == 0 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))
	
estimates store n4
	
// POF	
	
sls mktshare primp_pct primp_avgamt primp_avgt ///
	if rebate_dummy == 1 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))
	
estimates store p1
	
sls mktshare primp_expamt primp_avgt ///
	if rebate_dummy == 1 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))

estimates store p2
	
sls mktshare primp_pct primp_avgamt all_avgt ///
	if rebate_dummy == 1 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))
	
estimates store p3

sls mktshare primp_expamt all_avgt ///
	if rebate_dummy == 1 & ordertype == "Market", ///
	init(search("off"), conv_maxiter(50))
	
estimates store p4


// RMSE

estimates restore n1
ereturn list rmse
estimates restore n2
ereturn list rmse
estimates restore n3
ereturn list rmse
estimates restore n4
ereturn list rmse
estimates restore p1
ereturn list rmse
estimates restore p2
ereturn list rmse
estimates restore p3
ereturn list rmse
estimates restore p4
ereturn list rmse






