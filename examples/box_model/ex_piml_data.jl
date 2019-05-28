import FiniteVolume
import GaussianRandomFields
import JLD
import Random

include("plottools.jl")

Random.seed!(0)

sidelength = 50.0#m
#ns = [200, 150, 100]
ns_big = [100, 75, 50]
ns_small = [20, 15, 10]
mins = [-4 * sidelength, -3 * sidelength, -2 * sidelength]
maxs = [4 * sidelength, 3 * sidelength, 2 * sidelength]
meanloghyco = log(1e-5)#m/s
lefthead = 1.0#m
righthead = 0.0#m
#coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)

xs = range(mins[1]; stop=maxs[1], length=ns_big[1])
ys = range(mins[2]; stop=maxs[2], length=ns_big[2])
zs = range(mins[3]; stop=maxs[3], length=ns_big[3])
grf = GaussianRandomFields.GaussianRandomField(GaussianRandomFields.CovarianceFunction(3, GaussianRandomFields.Matern(30.0, 2.0)), GaussianRandomFields.CirculantEmbedding(), zs, ys, xs)
function samplehyco()
	return 3 * GaussianRandomFields.sample(grf) .+ meanloghyco
end

function gethead(p, ns)
	coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)
	dirichletnodes = Int[]
	dirichletheads = Float64[]
	for i = 1:size(coords, 2)
		if coords[1, i] == mins[1]
			push!(dirichletnodes, i)
			push!(dirichletheads, lefthead)
		elseif coords[1, i] == maxs[1]
			push!(dirichletnodes, i)
			push!(dirichletheads, righthead)
		end
	end
	loghycos = reshape(p, ns[3], ns[2], ns[1])
	neighborhycos = FiniteVolume.nodehycos2neighborhycos(neighbors, loghycos, true)
	sources = zeros(size(coords, 2))
	head, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, neighborhycos, sources, dirichletnodes, dirichletheads)
	if !ch.isconverged
		error("didn't converge")
	end
	if maximum(head) > 1 || minimum(head) < 0
		error("problem with solution -- head out of range")
	end
	return head
end

function coarsen(p, ns_big, ns_small)
	p_big = reshape(p, reverse(ns_big)...)
	p_small = zeros(reverse(ns_small)...)
	for i1 = 1:ns_big[3], i2 = 1:ns_big[2], i3 = 1:ns_big[1]
		i1_small = 1 + div(i1 - 1, round(Int, ns_big[3] / ns_small[3]))
		i2_small = 1 + div(i2 - 1, round(Int, ns_big[2] / ns_small[2]))
		i3_small = 1 + div(i3 - 1, round(Int, ns_big[1] / ns_small[1]))
		p_small[i1_small, i2_small, i3_small] += p_big[i1, i2, i3]
	end
	return p_small[:]
end

function samplepair()
	p = samplehyco()
	head_big = gethead(p, ns_big)
	p2 = coarsen(p, ns_big, ns_small)
	head_small = gethead(p2, ns_small)
	return reshape(p, reverse(ns_big)...), reshape(head_big, reverse(ns_big)...), reshape(p2, reverse(ns_small)...), reshape(head_small, reverse(ns_small)...)
end

numsamples = 10^1
allloghycos_big = Array{Float64}(undef, numsamples, reverse(ns_big)...)
allheads_big = Array{Float64}(undef, numsamples, reverse(ns_big)...)
allloghycos_small = Array{Float64}(undef, numsamples, reverse(ns_small)...)
allheads_small = Array{Float64}(undef, numsamples, reverse(ns_small)...)
ns = ns_big
@time for i = 1:numsamples
	global ns
	p_big, head_big, p_small, head_small = samplepair()
	allloghycos_big[i, :, :, :] = p_big
	allheads_big[i, :, :, :] = head_big
	allloghycos_small[i, :, :, :] = p_small
	allheads_small[i, :, :, :] = head_small
	ns = ns_big
	plotlayers([1, div(ns[3], 2), ns[3]], p_big[:])
	plotlayers([1, div(ns[3], 2), ns[3]], head_big[:])
	ns = ns_small
	plotlayers([1, div(ns[3], 2), ns[3]], p_small[:])
	plotlayers([1, div(ns[3], 2), ns[3]], head_small[:])
end
@time JLD.save("piml_data.jld", "allloghycos_big", allloghycos_big, "allheads_big", allheads_big, "allloghycos_small", allloghycos_small, "allheads_small", allheads_small)
