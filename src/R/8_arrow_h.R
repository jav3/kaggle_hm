# Helpers -----------------------------------------------------------------

Mode <- function(x) {
  if ( length(x) <= 2 ) return(x[1])
  if ( anyNA(x) ) x = x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

checkPR <- function(candidates, curr_yr = 2020, curr_wk, cids = NULL){
  transactions = open_dataset("data/transactions_p/", format = "arrow")
  
  if (is.null(cids)){
    curr_purchases = transactions %>% 
      filter(yr == curr_yr, wk == curr_wk) %>%
      select(cid, aid, t_dat) %>%
      collect()
  } else{
    curr_purchases = transactions %>% 
      filter(yr == curr_yr, wk == curr_wk, cid %in% cids) %>%
      select(cid, aid, t_dat) %>%
      collect()
  }
  setDT(curr_purchases)
  setorder(curr_purchases, cid, t_dat)
  curr_purchases = unique(curr_purchases, by = c("cid", "aid"))
  curr_purchases[, purchased := 1]
  
  candidates_df = candidates %>% mutate(candidate = 1) %>% as.data.table %>% unique(by = c("cid", "aid"))
  gc()
  
  both = merge(curr_purchases, 
               candidates_df[, .(aid, cid, candidate)],
               by = c("cid", "aid"), all.x = F, all.y = F)
  
  both_count = both[, .(both_count = min(12, .N)), by = .(cid)]
  curr_count = curr_purchases[, .(curr_count = min(12, .N)), by = .(cid)]
  curr_count = merge(curr_count, both_count, by = "cid", all.x = T, all.y = F)
  curr_count[is.na(curr_count), curr_count := 0]
  curr_count[is.na(both_count), both_count := 0]
  curr_count[, ap12 := both_count/curr_count]
  max_map12 = mean(curr_count$ap12)
  
  full_covered_cust = sum(curr_count$ap12 == 1)
  not_covered_cust = sum(curr_count$ap12 == 0)
  num_target_cust = nrow(curr_count)
  
  num_candidate = nrow(candidates_df)
  num_target = nrow(curr_purchases)
  num_unique_a = length(unique(candidates_df$aid))
  
  
  cat(sprintf("MAX MAP12: %s\n", round(max_map12, 4)))
  cat(sprintf("Full Covered Customer: %s\n", round(full_covered_cust / num_target_cust, 4)))
  cat(sprintf("Not Covered Customer: %s\n", round(not_covered_cust / num_target_cust, 4)))
  cat(sprintf("Candidate multiplier: %s\n", round(num_candidate / num_target, 4)))
  cat(sprintf("Unique Article Id: %s\n", num_unique_a))
  
  # Precision
  precision = num_candidate/sum(both$purchased, na.rm = T)
  
  # Recall
  recall = sum(both$candidate, na.rm = T)/num_target
  
  fscore = (2 * 1/precision * recall) / (1/precision + recall)
  
  cat("Precision: ", precision, "\nRecall: ",  recall, "\nFscore: ", fscore, "\n")
  return("_")
}


# Strategy ----------------------------------------------------------------
# Prior history is 36
# Train on 37
# Predict on 38

# Combine different strategies
executeStrategy <- function(curr_yr = 2020, curr_wk = 38, topx = 5, wks = 36:37, cids){
  
  # Prior purchases - s1
  week_end_date = as.Date("2020-01-01") %m+% days((curr_wk-1)*7 - 1)
  
  prior_purchases = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk & wk >= curr_wk - 6, cid %in% cids) %>%
    select(cid, aid, t_dat) %>%
    collect %>%
    as.data.table
  prior_purchases = merge(prior_purchases, articles_data[, .(aid, grp)], by = "aid")
  
  # Remove already re-purchased cases (exact or same group)
  repurchases = unique(prior_purchases, by = c("cid", "aid", "t_dat"))
  repurchases = repurchases[, num_purchases := .N, by = .(cid, aid)][order(cid, aid, t_dat)]
  already_repurchased = repurchases[num_purchases > 1, .(cid, aid)] %>% unique %>%
    mutate(repurchased = 1)
  prior_purchases = merge(prior_purchases, already_repurchased, by = c("cid", "aid"), all.x = T) %>%
    filter(is.na(repurchased)) %>%
    select(-repurchased)
  
  repurchases = unique(prior_purchases, by = c("cid", "grp", "t_dat"))
  repurchases = repurchases[, num_purchases := .N, by = .(cid, grp)][order(cid, grp, t_dat)]
  already_repurchased = repurchases[num_purchases > 1, .(cid, grp)] %>% unique %>%
    mutate(repurchased = 1)
  prior_purchases = merge(prior_purchases, already_repurchased, by = c("cid", "grp"), all.x = T) %>%
    filter(is.na(repurchased))
  
  # Remove cases where either customer does not return anything or article doesn't typically get returned
  
  
  prior_purchases = prior_purchases[order(cid, aid, desc(t_dat))] %>%
    unique(by = c("cid", "aid")) %>%
    merge(customers_dy[wk_predict == curr_wk, .(cid, hist_repurchase_c = hist_repurchase)], by = "cid") %>%
    merge(articles_dy[wk_predict == curr_wk, .(aid, hist_repurchase)], by = "aid") %>%
    # Label for strategy
    mutate(strat = "s1_prior_purchases", score = (1/as.numeric(week_end_date - t_dat + 1))*(hist_repurchase_c + hist_repurchase)) %>%
    select(-hist_repurchase_c, -hist_repurchase) %>%
    # Take topx
    as.data.table %>%
    {.[, head(.SD, topx), by = .(cid)]}
  
  # Similar items
  articles_similarity = articles_similarity_o[linked_aid %in% unique(prior_purchases$aid)]
  similar_purchase = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk, cid %in% cids) %>%
    select(cid, aid) %>%
    collect() %>%
    as.data.table %>%
    unique %>%
    merge(articles_similarity[score >= 0.8], by = "aid", all = F, allow.cartesian = T) %>%
    select(-aid) %>%
    rename(aid = linked_aid) %>%
    as.data.table() %>%
    {.[, .(score = sum(score)), by = .(cid, aid)]} %>%
    # Label for strategy
    mutate(strat = "s3_art_similarity") %>%
    # Take topx
    as.data.table %>%
    {.[order(cid, -score)]} %>%
    {.[, head(.SD, topx), by = .(cid)]}
  
  # Special offers
  # spec_offers = articles_dy %>%
  #   filter(wk_predict == curr_wk, popularity > 0, popularity > popularity_1d_prior) %>%
  #   filter(aid %in% articles_data[department_name == "Campaigns" &
  #                                   garment_group_name == "Special Offers"]$aid) %>%
  #   pull(aid)
  # spec_offers = cids %>%
  #   CJ(spec_offers) %>%
  #   setNames(c("cid", "aid"))  %>%
  #   # Label for strategy
  #   mutate(strat = "s4_spec_offers")
  # spec_offers = merge(spec_offers, articles_dy[wk_predict == curr_wk, .(aid, score = price)],
  #                     by = "aid", all.x = T)
  # spec_offers[is.na(score), score := max(articles_dy[wk_predict == curr_wk]$price, na.rm = T)]
  
  # SAR
  sar_curr = sar_all[wk_predict == curr_wk & cid %in% cids, 
                     .(cid, aid, score, strat = "s5_sar")]

  # MF
  # mf = readRDS("data/t8/mf.rds")
  # mf_curr = mf %>%
  #   filter(wk_predict == curr_wk, cid %in% cids) %>%
  #   filter(observed == 1) %>%
  #   select(cid, aid, score) %>%
  #   # Remove prior purchases
  #   anti_join(prior_purchases, by = c("cid", "aid")) %>%
  #   # Label for strategy
  #   mutate(strat = "s6_mf")
  # mf_new = mf %>%
  #   filter(wk_predict == curr_wk, cid %in% cids) %>%
  #   filter(observed == 0) %>%
  #   select(cid, aid, score) %>%
  #   mutate(strat = "s6_mf")
  # setDT(mf_new)
  # mf_new = mf_new[order(cid, -score)][, head(.SD, 3), by = .(cid)]
  # mf_all = rbindlist(list(mf_curr, mf_new)) %>% unique
  
  # Similar items - doc2vec

  articles_d2v = articles_d2v_o[order(aid, -score)][, head(.SD, topx), by = .(aid)]
  articles_d2v = articles_d2v[linked_aid %in% unique(prior_purchases$aid)][, .(aid, linked_aid, score)]
  similar_purchase_d2v = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk, cid %in% cids) %>%
    select(cid, aid) %>%
    collect() %>%
    as.data.table %>%
    unique %>%
    merge(articles_d2v, by = "aid", all = F, allow.cartesian = T) %>%
    select(-aid) %>%
    rename(aid = linked_aid) %>%
    as.data.table() %>%
    {.[, .(score = sum(score)), by = .(cid, aid)]} %>%
    # Label for strategy
    mutate(strat = "s7_art_d2v") %>%
    # Take topx
    as.data.table %>%
    {.[order(cid, -score)]} %>%
    {.[, head(.SD, topx), by = .(cid)]}
  
  # # Popular cases by price category
  # users_binned = transactions %>%
  #   filter(yr == curr_yr, wk < curr_wk & wk %in% wks, cid %in% cids) %>%
  #   select(cid, aid, price) %>%
  #   collect
  # setDT(users_binned)
  # users_binned = users_binned[, .(price_binned = round(mean(price, na.rm = T)/10)*10), by = .(cid)]
  # articles_binned = articles_dy[wk_predict == curr_wk & !is.na(price),
  #                               .(aid, popularity_num, score = price, price_binned = round(price/10)*10)]
  # articles_binned = articles_binned[order(price_binned, popularity_num)][, head(.SD, 10), by = .(price_binned)]
  # users_binned_2 = users_binned %>%
  #   merge(articles_binned, by = "price_binned", allow.cartesian = T) %>%
  #   # Label for strategy
  #   mutate(strat = "s8_price_bin") %>%
  #   select(cid, aid, score, strat)

  # Bought together items
  tgr_purchases = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk, cid %in% cids) %>%
    select(cid, aid) %>%
    collect() %>%
    as.data.table %>%
    unique
  tgr_purchases = tgr_purchases %>%
    merge(btgr[wk_predict == curr_wk], by.x = "aid", by.y = "aidX", all = F, allow.cartesian = T) %>%
    {.[, .(score = sum(wt)), by = .(cid, aidY)]} %>%
    merge(tgr_purchases %>% mutate(prior_purchase = 1), 
          by.x = c("cid", "aidY"), by.y = c("cid", "aid"), all.x = T, all.y = F) %>%
    filter(is.na(prior_purchase)) %>%
    select(-prior_purchase) %>%
    rename(aid = aidY) %>%
    # Label for strategy
    mutate(strat = "s9_btgr")
  tgr_purchases = tgr_purchases[order(cid, -score)][, head(.SD, min(.N, topx)), by = .(cid)]
  
  articles_fse = fse[order(aid, -scores)][, head(.SD, topx), by = .(aid)]
  articles_fse = articles_fse[linked_aid %in% unique(prior_purchases$aid)][, .(aid, linked_aid, score = scores)]
  similar_purchase_fse = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk, cid %in% cids) %>%
    select(cid, aid) %>%
    collect() %>%
    as.data.table %>%
    unique %>%
    merge(articles_fse, by = "aid", all = F, allow.cartesian = T) %>%
    select(-aid) %>%
    rename(aid = linked_aid) %>%
    as.data.table() %>%
    {.[, .(score = sum(score)), by = .(cid, aid)]} %>%
    # Label for strategy
    mutate(strat = "s10_fse") %>%
    # Take topx
    as.data.table %>%
    {.[order(cid, -score)]} %>%
    {.[, head(.SD, topx), by = .(cid)]}
  
  articles_pe = prodemb[order(aid, -scores)][, head(.SD, topx), by = .(aid)]
  articles_pe = articles_pe[linked_aid %in% unique(prior_purchases$aid)][, .(aid, linked_aid, score = scores)]
  similar_purchase_pe = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk & wk %in% max(wks), cid %in% cids) %>%
    select(cid, aid) %>%
    collect() %>%
    as.data.table %>%
    unique %>%
    merge(articles_pe, by = "aid", all = F, allow.cartesian = T) %>%
    select(-aid) %>%
    rename(aid = linked_aid) %>%
    as.data.table() %>%
    {.[, .(score = sum(score)), by = .(cid, aid)]} %>%
    # Label for strategy
    mutate(strat = "s11_prodemb") %>%
    # Take topx
    as.data.table %>%
    {.[order(cid, -score)]} %>%
    {.[, head(.SD, topx), by = .(cid)]}
  
  # Top popular - s2
  top_pop_last_w = articles_dy[wk_predict == curr_wk & popularity_num <= topx, aid]
  top_pop_last_w = cids %>%
    CJ(top_pop_last_w) %>%
    setNames(c("cid", "aid")) %>%
    # Label for strategy
    mutate(strat = "s2_top_popular")
  top_pop_last_w = merge(top_pop_last_w, articles_dy[wk_predict == curr_wk, .(aid, score = price)], 
                         by = "aid", all.x = T)
  top_pop_last_w[is.na(score), score := max(articles_dy[wk_predict == curr_wk]$price, na.rm = T)]
  
  all = rbindlist(list(
    prior_purchases,
    similar_purchase,
    # spec_offers,
    sar_curr,
    similar_purchase_d2v,
    # users_binned_2,
    # mf_all,
    tgr_purchases,
    similar_purchase_fse,
    similar_purchase_pe,
    top_pop_last_w
  ), use.names = T, fill = T)
  
  all = dcast.data.table(all, cid + aid ~ strat, value.var = "score", fill = 0)
  
  all[, num_strats_hit := Reduce(`+`, lapply(.SD, function(x) replace(x, x != 0, 1))), .SDcols = 3:ncol(all)]

  setDT(all)
  
  return(all)
}

# Construct candidates -------------------------------------------------

constructCandidates <- function(negatives, purchased_customers, curr_yr = 2020, curr_wk = 38, wks = 36:37, keep_purchased = F){
  
  # Merge purchase indicator
  if (!keep_purchased){ # Do not keep purchased when measuring on validation
    candidates = merge(negatives, purchased_customers, 
                       by = c("aid", "cid"), all.x = T, all.y = F) 
  } else{
    candidates = merge(negatives, purchased_customers, 
                       by = c("aid", "cid"), all.x = T, all.y = T) 
  }
  num_purchased_in_strategy = length(unique(candidates[purchased == 1]$cid))
  
  candidates[is.na(purchased), purchased := 0]
  
  if (length(unique(purchased_customers$cid)) > 0)
  cat(sprintf("\nINFO: %s customers purchased via strategy out of %s who purchased in total\n", 
              num_purchased_in_strategy, length(unique(purchased_customers$cid))))
  
  # Add dynamic customer attr
  candidates = merge(candidates, 
                     customers_dy[wk_predict == curr_wk,
                                  .(cid, mean_purchases_cw_ch1, mean_purchases_cw_ch2, modal_index_code, 
                                  modal_index_group_no, modal_garment_group_no, 
                                  average_channel, average_price, average_days_on_market, 
                                  average_all_time_lowest, average_popularity, days_since_last_seen_c,
                                  hist_repurchase_c = hist_repurchase, hist_repurchase_same_grp_c = hist_repurchase_same_grp)],
                     by = "cid", all.x = T)  
  
  # Add static customer attr
  candidates = merge(candidates, 
                     customers %>% select(cid, age, Active) %>% mutate(Active = as.numeric(Active)) %>%
                       as.data.table, 
                     by = "cid", all.x = T)

  # Add dynamic article attr
  candidates = merge(candidates, 
                     articles_dy[wk_predict == curr_wk,
                                 .(aid, popularity, popularity_1d_prior, popularity_num_1d_prior, days_on_market,
                                   new_arrival_1w, new_arrival_2w, new_arrival_1m, days_since_last_seen, price, 
                                   md_lw, md_lm, md_l2m, all_time_lowest, discount, old_version, stock_count,
                                   ttnp, hist_repurchase, hist_repurchase_same_grp)], 
                     by = "aid", all.x = T)
  
  # Add static article attr
  candidates = merge(candidates, 
                     articles %>% select(aid, index_group_no, index_code, product_group_name, 
                                         perceived_colour_value_name, garment_group_no) %>% collect %>%
                       mutate(index_group_no = as.numeric(index_group_no),
                              index_code = as.numeric(index_code),
                              product_group_name = as.numeric(product_group_name),
                              perceived_colour_value_name = as.numeric(perceived_colour_value_name),
                              garment_group_no = as.numeric(garment_group_no)) %>%
                       as.data.table, 
                     by = "aid", all.x = T)
  
  # Create same_x variables
  candidates[, abs_average_price :=  abs(average_price - price)]
  candidates[, abs_average_days_on_market :=  abs(average_days_on_market - days_on_market)]
  candidates[, abs_average_all_time_lowest :=  abs(average_all_time_lowest - as.numeric(all_time_lowest))]
  candidates[, abs_average_popularity :=  abs(average_popularity - popularity)]
  candidates[, same_index_code := modal_index_code == index_code]
  candidates[, same_index_group_no := modal_index_group_no == index_group_no]
  candidates[, same_garment_group_no := modal_garment_group_no == garment_group_no]
    
  # Add similarity measures
  # Get recent purchases
  recent_purchases = transactions %>% 
    filter(yr == curr_yr, wk < curr_wk, wk %in% wks, cid %in% unique(candidates$cid)) %>%
    select(cid, aid, t_dat) %>%
    distinct %>%
    collect
  setDT(recent_purchases)
  setorder(recent_purchases, cid, aid, -t_dat)
  recent_purchases = unique(recent_purchases, by = c("cid", "aid"))
  recent_purchases = recent_purchases[, .SD[1:min(.N, 12)], by = .(cid)]
  
  aid_similarity = merge(candidates[, .(cid, aid)], 
                         recent_purchases[, .(cid, linked_aid = aid)], by = "cid", all.x = F, all.y = F, allow.cartesian = T) %>%
    merge(articles_similarity_o, by =c("aid", "linked_aid"), all.x = F) 
  setDT(aid_similarity)
  aid_similarity = aid_similarity[, .(similarity = max(score, na.rm = T)), by = .(cid, aid)]
  candidates = merge(candidates, aid_similarity, by = c("cid", "aid"), all.x = T)
  candidates[is.na(similarity), similarity := 0]
  
  gc()
  
  setDT(candidates)
  return(candidates)
}

validateModel <- function(candidates, model, curr_yr = 2020, curr_wk = 38, cids, cols = NULL, model.xgb = NULL){
  # Check MAP on entire candidate set for those who purchased
  
  if (!is.null(cols)){
    x = cols
  }
  
  # Get actual purchases
  actuals = transactions %>% filter(yr == curr_yr, wk == curr_wk, cid %in% cids) %>% 
    select(aid, cid) %>%
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
  if (!is.null(model.xgb)){
    to_validate[, pred2 := {setTxtProgressBar(pb, .GRP); predict(model.xgb, .SD[, x, with = F] %>% sapply(as.numeric) %>% as.matrix);}, by = .(grp)]
    to_validate[, pred := (pred + pred2)/2]
  }
  cat("\nFinished computing predictions\n")
  
  if (0){
    to_validate_2 = merge(to_validate, actuals[, .(aid = unlist(actual), actual = 1), by = .(cid)], by = c("cid", "aid"), all = T)
    to_validate_2[, missed_candidate := is.na(pred)]
    to_validate_2[, actual_purchase := !is.na(actual)]
    to_validate_2[, table(missed_candidate, actual_purchase)]
    to_validate_2[actual_purchase == T & missed_candidate == T][, .(.N), by = .(aid)][order(-N)]
    transactions %>% filter(cid == 1993) %>% collect %>% merge(articles_data, by = "aid") %>% arrange(yr, wk) %>% View
  }
  
  to_validate = to_validate[order(cid, -pred)][, .(aid = list(aid[1:min(12, .N)])), by = .(cid)]
  to_validate = merge(to_validate, actuals, by = "cid", all = T)
  
  mapk = mapk(12, actual = to_validate$actual, predicted = to_validate$aid)
  cat("Map@12 is:", mapk, "\n")
  return(mapk)
}
