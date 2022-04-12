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
using Flux
using LinearAlgebra
include("src/helpers.jl")

# Import data
transactions = DataFrame(Arrow.Table("data/transactions_m.arrow"))

customers = DataFrame(Arrow.Table("data/customers_m.arrow"))

articles = DataFrame(Arrow.Table("data/articles_m.arrow"))

# Filter on eight weeks of training period
trans = filter(a -> a.t_dat > Date(2020,09,22) - Day(7*2), transactions, view = true)

num_customers = length(unique(trans.cid))
num_articles = length(unique(trans.aid))

customer_trans_count = @pipe combine(groupby(trans, :cid), nrow) |> filter(:nrow => x -> x > 30, _)

trans = filter(:cid => in(customer_trans_count.cid), trans)

# Matrix of customers x articles
num_customers = length(unique(trans.cid))
num_articles = length(unique(trans.aid))
customers_dict = Dict(val => idx for (idx, val) in enumerate(unique(trans.cid)))
articles_dict = Dict(val => idx for (idx, val) in enumerate(unique(trans.aid)))

# Index of articles in subset
trans[:, :cidx] = getindex.(Ref(customers_dict), trans[:, :cid])
trans[:, :aidx] = getindex.(Ref(articles_dict), trans[:, :aid]) 

t_mat = spzeros(num_articles, num_customers)
t_mat[CartesianIndex.(trans.aidx, trans.cidx)] .= 1

# Layers: (num_articles x 5) * (5 x num_articles)
W = Chain(Dense(num_articles, 5), Dense(5, num_articles))

# Loss function
loss(x, y) = Flux.crossentropy(W(x), y)

# Simple measure
loss(trn_features, trn_labels)

# Optimizer
optimizer = Descent(0.01)

Flux.@epochs 100 Flux.train!(loss, params(W), data, optimizer)
loss(trn_features, trn_labels)

@pipe StatsBase.countmap(Flux.onecold(W(trn_features))) |> DataStructures.sort(_, byvalue = true, rev = true)
@pipe StatsBase.countmap(Flux.onecold(trn_labels)) |> DataStructures.sort(_, byvalue = true, rev = true)

mean(Flux.onecold(W(trn_features)) .== Flux.onecold(trn_labels))

U = rand(num_customers, 5)
V = rand(5, num_articles)

function loss(y)
    sum((y .- m()).^2) #-sum(y .* (U*V) .- log.(exp.(U*V) .+ 1))
end

new_loss = loss(t_mat, U, V)
curr_loss = new_loss + thresh + 1.0
thresh = 10.0

while (curr_loss - new_loss > thresh)
    curr_loss = new_loss
    gs = gradient(() -> loss(t_mat), params(U))
    U .-= 0.0001 .* gs[U]
    gs = gradient(() -> loss(t_mat), params(V))
    V .-= 0.001 .* gs[V]
    new_loss = loss(t_mat)
    diff = curr_loss - new_loss
    println("New loss: $new_loss")
end

t_mat_d = svd(Matrix(t_mat))