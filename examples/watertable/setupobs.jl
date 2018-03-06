import FEHM
import JLD

meshdir = "mesh25"
#meshdir = "mesh12.5"
#meshdir = "mesh6.25"
#meshdir = "mesh3.125"
obsdata = readdlm(joinpath(dirname(@__FILE__), "data", "gooddata.dat"))
coords = FEHM.parsegrid(joinpath(dirname(@__FILE__), meshdir, "wtr25.fehmn"))
xs = coords[1, :]
ys = coords[2, :]
zs = coords[3, :]
rechargenodes = JLD.load("$meshdir.jld", "rechargenodes")
obsnodes = Int[]
obsvalues = Float64[]
function findnearestnode(x, y, xs, ys, nodes)
	bestnode = -1
	bestdist = Inf
	for node in nodes
		thisdist = sqrt((xs[node] - x)^2 + (ys[node] - y)^2)
		if thisdist < bestdist
			bestdist = thisdist
			bestnode = node
		end
	end
	return bestnode
end
for i = 1:size(obsdata, 1)
	bestnode = findnearestnode(obsdata[i, 2], obsdata[i, 3], xs, ys, rechargenodes)
	push!(obsnodes, bestnode)
	push!(obsvalues, obsdata[i, 4])
end
tophyconodes, hycoxs, hycoys = JLD.load("$meshdir.jld", "tophyconodes", "hycoxs", "hycoys")
hycoobsnodes = Int[]
hycoobsvalues = Float64[]
hycodata = readdlm(joinpath(dirname(@__FILE__), "data", "hyco_meters_per_day.dat"))
for i = 1:size(hycodata, 1)
	bestnode = findnearestnode(hycodata[i, 2], hycodata[i, 3], hycoxs, hycoys, tophyconodes)
	push!(hycoobsnodes, bestnode)
	push!(hycoobsvalues, hycodata[i, 6] / (60 * 60 * 24))#convert from m/day to m/s
end
A = hcat(ones(length(obsvalues)), Float64.(obsdata[:, 2]), Float64.(obsdata[:, 3]))
b = obsvalues
coeffs = A \ b
dirichletnodes = JLD.load("$meshdir.jld", "dirichletnodes")
dirichletheads0 = Array{Float64}(length(dirichletnodes))
for (i, node) in enumerate(dirichletnodes)
	dirichletheads0[i] = dot(coeffs, [1, xs[node], ys[node]])
end
JLD.save("observations_$meshdir.jld", "obsnodes", obsnodes, "obsvalues", obsvalues, "dirichletheads0", dirichletheads0, "obsxs", Float64.(obsdata[:, 2]), "obsys", Float64.(obsdata[:, 3]), "obsnames", obsdata[:, 1], "hycoobsnodes", hycoobsnodes, "hycoobsvalues", hycoobsvalues)
