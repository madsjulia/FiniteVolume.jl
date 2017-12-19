import FiniteVolume
import JLD
import PyAMG

meshdirs = ["fourfractures", "circuit", "backbone_x01", "homogenous-10m", "pl_alpha_1.6", "25L_network_x2"]
#meshdirs = ["circuit"]
for meshdir in meshdirs
	@show meshdir
	@time xs, ys, zs, neighbors, areasoverlengths, fractureindices, dirichletnodes, dirichletheads, conductivities = JLD.load(joinpath(meshdir, "mesh.jld"), "xs", "ys", "zs", "neighbors", "areasoverlengths", "fractureindices", "dirichletnodes", "dirichletheads", "conductivities")
	sources = zeros(length(xs))
	node2fracture = Dict(zip(1:length(xs), fractureindices))
	connection2fracture = Dict(zip(1:length(conductivities), map(p->node2fracture[p[1]], neighbors)))
	metaindex(i) = connection2fracture[i]
	t = @elapsed @time h, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads)
	@show length(xs), length(xs) / t
end
