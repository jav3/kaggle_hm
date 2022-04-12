# Working dir
cd("/Users/javidali/Downloads/Kaggle/hm")

# Packages
using CSV
using DataFrames
using Dates
using Pipe
using Query
using Arrow
using Distributed
using ProgressMeter
using Statistics
using SparseArrays
include("src/helpers.jl")

# Import data
transactions = DataFrame(Arrow.Table("data/transactions.arrow"))

customers = DataFrame(Arrow.Table("data/customers.arrow"))
customers_set = Set(customers.customer_id);

articles = DataFrame(Arrow.Table("data/articles.arrow"))

# Filter on eight weeks of training period
trans = filter(a -> a.t_dat > Date(2020,09,22) - Day(7*8), transactions, view = true)

# Create weekly groups
trans.week = Week.(Date.(trans.t_dat) - Day(2))


# Create dictionary for customers
weeks = @pipe unique(trans.week) |> sort |> Dict(_[i] => i for i in 1:length(_))
weeks_list = collect(keys(weeks))

# Create an dict to store actual purchases in a given week
trans_group = groupby(trans, [:customer_id, :week])
actuals = Dict(key => unique(val.article_id) for (key, val) in pairs(trans_group))

# Strategy 1 - predict last 12 purchases as purchases for next week
s1 = Dict(name => Vector{Union{Missing, Float16}}() for name in customers.customer_id);

trans_group = groupby(trans, :customer_id);

@showprogress for (key, sub_df) in pairs(trans_group)

    preds = Vector{Union{Missing, String}}(missing, 12);

    res = Vector{Union{Missing, Float16}}(missing, length(weeks) - 1)

    for w in 1:(length(weeks) - 1)
        # Get articles purchased in the current week
        curr_articles = filter(:week => in(Set([weeks_list[w]])), sub_df).article_id;
        # Add to predicted array
        for c in curr_articles
            helpers.update_vec!(preds, c);
        end

        check_key = haskey(actuals, (customer_id = key.customer_id, week = weeks_list[w+1]))

        if (check_key)
            curr_actual = actuals[(customer_id = key.customer_id, week = weeks_list[w+1])]
        else
            curr_actual = []
        end
        res[w] = helpers.apk(curr_actual, preds);
    end
    setindex!(s1, res, key.customer_id)
end

# MAP@12 for each week
for i in 1:(length(weeks)-1)
    @pipe map(x -> if(isempty(x)) missing else x[i] end, values(s1)) |> (sum(skipmissing(_))/(sum(skipmissing(_) .>=0)*1.0)) |> print("MAP for week $i:$_\n")
end

#     MAP for week 1:0.006340426559744933
#     MAP for week 2:0.0011393655065854236
#     MAP for week 3:0.0065416369919020865
#     MAP for week 4:0.009023391406954953
#     MAP for week 5:0.013503249219343405
#     MAP for week 6:0.0033939029340136844
#     MAP for week 7:0.01168441661228287

# Strategy 2 - one prior prediction
# P(item 2 | item 1)

# Filter on 8 weeks of training period
trans = filter(a -> a.t_dat > Date(2020,09,22) - Day(7*8), transactions, view = true)


# Create a sparse one-step NxN matrix, where N = articles
num_articles = nrow(articles)
t_mat = spzeros(num_articles,num_articles)

# Iterate on transactions now and add 1 to the article index
articles_set = @pipe articles.article_id |> Dict(_[i] => i for i in 1:length(_))

trans_group = groupby(trans[:, [:article_id, :customer_id]], :customer_id);

@showprogress for (key, sub_df) in pairs(trans_group)

    # Convert article id to index
    idx = getindex.(Ref(articles_set), sub_df.article_id)

    if (length(idx) > 1)
        t_mat[CartesianIndex.(idx[1:(length(idx)-1)], idx[2:length(idx)])] .+= 1
    end
end

# Array too sparse -- aborting

# Strategy 3 - simple popularity

# Filter on eight weeks of training period
trans = filter(a -> a.t_dat > Date(2020,09,22) - Day(7*4), transactions, view = true)

# Popular articles
popular_articles = @pipe combine(groupby(trans, :article_id), nrow) |> sort(_, :nrow, rev = true) |> filter(:nrow => x -> x > 300, _)
# 708

# Join article metadata
leftjoin!(popular_articles, articles, on = :article_id)

# View
vcat(
    mapcols(x -> length(unique(x)), popular_articles), 
    mapcols(x -> length(unique(x)), articles),
    cols = :union) |> vscodedisplay

# Filter on one week of training period
trans = filter(a -> a.t_dat > Date(2020,09,22) - Day(7), transactions, view = true)

# Popular articles
popular_articles = @pipe combine(groupby(trans, :article_id), nrow) |> sort(_, :nrow, rev = true) #|> filter(:nrow => x -> x > 300, _)
# 46

# Number of customers
length(unique(trans.customer_id))
# 68984

# Take top 12 and check MAP@12 on it
preds = popular_articles[1:12, :article_id]

filter(:article_id => in(preds), trans)

s3 = Dict{String, Union{Missing, Float16}}()
for name in customers.customer_id
    s3[name] = missing
end

trans_group = groupby(trans, :customer_id);

@showprogress for (key, sub_df) in pairs(trans_group)
    curr_articles = sub_df.article_id |> unique;
    res = helpers.apk(curr_articles, preds);
    setindex!(s3, res, key.customer_id)
end

# MAP@12 
@pipe values(s3) |> (sum(skipmissing(_))/(sum(skipmissing(_) .>=0)*1.0)) |> print("MAP :$_\n")
# MAP :0.00869766902470138
