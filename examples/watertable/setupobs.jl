import FEHM
import JLD

#=
obsdata = readdlm(joinpath(dirname(@__FILE__), "data", "d2017May.dat"))
obsdata2= readdlm(joinpath(dirname(@__FILE__), "data", "i2015-2017May.dat"))
obsdata = vcat(obsdata, obsdata2)
=#
obsdata = readdlm(joinpath(dirname(@__FILE__), "data", "gooddata.dat"))
coords = FEHM.parsegrid(joinpath(dirname(@__FILE__), "data", "wtr25.fehmn"))
xs = coords[1, :]
ys = coords[2, :]
zs = coords[3, :]
rechargenodes = JLD.load("model.jld", "rechargenodes")
obsnodes = Int[]
obsvalues = Float64[]
for i = 1:size(obsdata, 1)
	bestnode = -1
	bestdist = Inf
	for node in rechargenodes
		thisdist = sqrt((xs[node] - obsdata[i, 2])^2 + (ys[node] - obsdata[i, 3])^2)
		if thisdist < bestdist
			bestdist = thisdist
			bestnode = node
		end
	end
	push!(obsnodes, bestnode)
	push!(obsvalues, obsdata[i, 4])
end
A = hcat(ones(length(obsvalues)), Float64.(obsdata[:, 2]), Float64.(obsdata[:, 3]))
b = obsvalues
coeffs = A \ b
dirichletnodes = JLD.load("model.jld", "dirichletnodes")
dirichletheads0 = Array{Float64}(length(dirichletnodes))
for (i, node) in enumerate(dirichletnodes)
	dirichletheads0[i] = dot(coeffs, [1, xs[node], ys[node]])
end
JLD.save("observations.jld", "obsnodes", obsnodes, "obsvalues", obsvalues, "dirichletheads0", dirichletheads0, "obsxs", Float64.(obsdata[:, 2]), "obsys", Float64.(obsdata[:, 3]), "obsnames", obsdata[:, 1])
