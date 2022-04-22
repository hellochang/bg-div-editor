## Queries for ```bg_div```
fsym_id is used as an argument.
```
	select * from fstest.dbo.bg_div where fsym_id ='XFBRC3-R'
```

```
	select max(exdate) from fstest.dbo.bg_div where fsym_id = 'XFBRC3-R'
```
```
	select 1 from fstest.dbo.bg_div where fsym_id = 'XFBRC3-R'
```

## Query for split
fsym_id is used as an argument.
```
select 
                fsym_id,p_split_date,p_split_factor, 
                exp(sum(log(p_split_factor))  OVER (ORDER BY p_split_date desc)) cum_split_factor 
                from fstest.fp_v2.fp_basic_splits where fsym_id= 'XFBRC3-R'
                order by p_split_date
```
## Query for basic info
fsym_id is used as an argument.
```
 select 
			 proper_name, currency, tr.ticker_region 'ticker'
            from fstest.sym_v1.sym_coverage sc
            left join fstest.sym_v1.sym_ticker_region tr on tr.fsym_id = sc.fsym_id
            where sc.fsym_id = 'XFBRC3-R'
```

## Query for new div data
fsym_id and p_divs_exdate are used as arguments.
```
	select 
			bd.fsym_id, bbg.bbg_id,currency,p_divs_exdate,p_divs_recdatec,p_divs_paydatec,
				p_divs_pd,p_divs_s_pd from fstest.fp_v2.fp_basic_dividends bd
              left join fstest.sym_v1.sym_bbg bbg on bbg.fsym_id = bd.fsym_id
			  where bd.fsym_id = 'XFBRC3-R' and p_divs_exdate > '2021-08-05'
```
