import FiniteVolume
import GaussianRandomFields
import JLD
import Random

include("plottools.jl")

Random.seed!(0)

sidelength = 50.0#m
thickness = 10.0#m
mins = [-sidelength, -sidelength, 0]
maxs = [sidelength, sidelength, thickness]
ns = [10, 10, 10]
meanloghyco = log(1e-5)#m/s
lefthead = 1.0#m
righthead = 0.0#m
coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)

xs = range(mins[1]; stop=maxs[1], length=ns[1])
ys = range(mins[2]; stop=maxs[2], length=ns[2])
zs = range(mins[3]; stop=maxs[3], length=ns[3])
grf = GaussianRandomFields.GaussianRandomField(GaussianRandomFields.CovarianceFunction(3, GaussianRandomFields.Matern(10.0, 2.0)), GaussianRandomFields.CirculantEmbedding(), zs, ys, xs)
function samplehyco()
	return 3 * GaussianRandomFields.sample(grf) .+ meanloghyco
end
dirichletnodes = Int[]
dirichletheads = Float64[]
for i = 1:size(coords, 2)
	if coords[1, i] == -sidelength
		push!(dirichletnodes, i)
		push!(dirichletheads, lefthead)
	elseif coords[1, i] == sidelength
		push!(dirichletnodes, i)
		push!(dirichletheads, righthead)
	end
end

function gethead(p)
	loghycos = reshape(p, ns[3], ns[2], ns[1])
	neighborhycos = FiniteVolume.nodehycos2neighborhycos(neighbors, loghycos, true)
	sources = zeros(size(coords, 2))
	head, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, neighborhycos, sources, dirichletnodes, dirichletheads)
	return head
end

function samplepair()
	p = samplehyco()
	head = gethead(p)
	return reshape(p, ns[3], ns[2], ns[1]), reshape(head, ns[3], ns[2], ns[1])
end

numsamples = 10^5
allloghycos = Array{Float64}(undef, numsamples, ns...)
allheads = Array{Float64}(undef, numsamples, ns...)
@time for i = 1:numsamples
	p, head = samplepair()
	allloghycos[i, :, :, :] = p
	allheads[i, :, :, :] = head
	#=
	plotlayers([1, 5, 10], p[:])
	plotlayers([1, 5, 10], head[:])
	=#
end
@time JLD.save("hydrology_data.jld", "allloghycos", allloghycos, "allheads", allheads)
