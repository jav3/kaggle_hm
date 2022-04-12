# Helpers -----------------------------------------------------------------

Mode <- function(x) {
  if ( length(x) <= 2 ) return(x[1])
  if ( anyNA(x) ) x = x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Strategy ----------------------------------------------------------------
# Prior history is 36
# Train on 37
# Predict on 38

# Strategy - prior purchases, backfilled with most popular items in prior week
executeStrategy <- function(curr_yr = 2020, curr_wk = 37, topx = 20, wks = wks_train){
  
  # Prior purchases - s1
  prior_purchases = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk & wk %in% wks) %>%
    select(cid, aid) %>%
    map_batches(function(batch){
      batch %>%
        distinct(cid, aid)
    }) %>%
    distinct(cid, aid) %>%
    # Label for strategy
    mutate(strat = "s1")
  
  # Top popular - s2
  top_pop_last_w = articles_dy %>%
    filter(wk == curr_wk - 1, popularity_num <= topx) %>%
    pull(aid)
  top_pop_last_w = customers %>% 
    select(cid) %>% 
    pull %>% 
    CJ(top_pop_last_w) %>%
    setNames(c("cid", "aid")) %>%
    # Label for strategy
    mutate(strat = "s2")
  
  all = rbindlist(list(
    prior_purchases,
    top_pop_last_w
  ), use.names = T, fill = T)
  all[, ind := 1]
  all = dcast.data.table(all, cid + aid ~ strat, value.var = "ind", fill = 0)
  
  setDT(all)
  
  return(all)
}

# Construct candidates -------------------------------------------------

constructCandidates <- function(negatives, curr_yr = 2020, curr_wk = 37){
  
  # Merge purchase indicator
  setDT(negatives)
  negatives_with_ind = merge(negatives, 
                             transactions %>% filter(yr == curr_yr, wk == curr_wk) %>% 
                               select(aid, cid) %>% distinct %>% mutate(purchased = 1), 
                             by = c("aid", "cid"), all.x = T, all.y = F) 
  
  candidates = negatives_with_ind
  setDT(candidates)
  
  candidates[is.na(purchased), purchased := 0]
  
  # Add static customer attr
  candidates = merge(candidates, 
                     customers %>% select(cid, age, Active) %>% mutate(Active = as.numeric(Active)), 
                     by = "cid", all.x = T)
  # Add dynamic customer attr
  candidates = merge(candidates, 
                     customers_dy %>% filter(wk == curr_wk) %>%
                       select(cid, num_purchases_cw_ch1, num_purchases_cw_ch2,
                              modal_index_code, modal_index_group_no, preferred_channel, days_since_last_seen_c),
                     by = "cid", all.x = T)
  # Add dynamic article attr
  candidates = merge(candidates, 
                     articles_dy %>% filter(wk == curr_wk) %>% select(aid, last_w_popularity, last2_w_popularity, days_on_market, days_since_last_seen), 
                     by = "aid", all.x = T)
  # Add static article attr
  candidates = merge(candidates, 
                     articles %>% select(aid, index_group_no, index_code) %>% collect %>%
                       mutate(index_group_no = as.numeric(index_group_no), index_code = as.numeric(index_code)), 
                     by = "aid", all.x = T)
  gc()
  
  setDT(candidates)
  return(candidates)
}

validateModel <- function(candidates, model, curr_yr = 2020, curr_wk = 37){
  # Check MAP on entire candidate set for those who purchased
  
  # Get actual purchases
  actuals = transactions %>% filter(yr == curr_yr, wk == curr_wk) %>% select(aid, cid) %>%
    collect %>% setDT %>% 
    {.[, .(actual = list(aid)), by = .(cid)]}
  
  to_validate = copy(candidates)
  setDT(to_validate)
  to_validate = to_validate[cid %in% unique(actuals$cid)]
  to_validate[, grp := floor(.I/100000)*100000]
  
  cat("Computing predictions\n")
  grpn = uniqueN(to_validate$grp)
  pb <- txtProgressBar(min = 0, max = grpn, style = 3)
  to_validate[, pred := {setTxtProgressBar(pb, .GRP); model$predict(.SD[, x, with = F] %>% as.matrix);}, by = .(grp)]
  cat("\nFinished computing predictions\n")
  
  to_validate = to_validate[order(cid, -pred)][, .(aid = list(aid[1:min(12, .N)])), by = .(cid)]
  to_validate = merge(to_validate, actuals, by = "cid", all = T)
  
  mapk(12, actual = to_validate$actual, predicted = to_validate$aid)
}
