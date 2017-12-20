import JLD

function badmesh2goodmesh(meshdir="fourfractures/")
	lines = readlines(joinpath(meshdir, "full_mesh_vol_area.uge"); chomp=false)
	numcells = parse(Int, split(lines[1])[2])
	numconnections = parse(Int, split(lines[numcells + 2])[2])
	cellfile = open(joinpath(meshdir, "cells.dlm"), "w")
	for i = 2:numcells + 1
		write(cellfile, lines[i])
		#write(cellfile, "\n")
	end
	close(cellfile)
	connectionfile = open(joinpath(meshdir, "connections.dlm"), "w")
	for i = numcells + 3:numcells + 3 + numconnections - 1
		write(connectionfile, lines[i])
		#write(connectionfile, "\n")
	end
	close(connectionfile)
end

function goodmesh2greatmesh(meshdir, isdirichletnode, dirichlethead)
	celldata::Array{Float64, 2} = readdlm(joinpath(meshdir, "cells.dlm"), Float64)
	connectiondata::Array{Float64, 2} = readdlm(joinpath(meshdir, "connections.dlm"), Float64)
	fractureconductivities::Array{Float64, 1} = readdlm(joinpath(meshdir, "perm.dat"), Float64; skipstart=1)[:, 4]
	local fractureindices::Array{Int, 1}
	try
		fractureindices = readdlm(joinpath(meshdir, "materialid.dat"), Int; skipstart=3)[:, 1]
	catch#sometimes there are two columns and the second column has float64's
		fractureindices = readdlm(joinpath(meshdir, "materialid.dat"), Float64; skipstart=4)[:, 1]
	end
	xs = celldata[:, 2]
	ys = celldata[:, 3]
	zs = celldata[:, 4]
	n1 = map(Int, connectiondata[:, 1])
	n2 = map(Int, connectiondata[:, 2])
	neighbors = map(Pair, n1, n2)
	conductivities = Array{Float64}(length(neighbors))
	for i = 1:length(n1)
		conductivities[i] = sqrt(fractureconductivities[fractureindices[n1[i]]] * fractureconductivities[fractureindices[n2[i]]])
	end
	areas = connectiondata[:, end]
	lengths = map((i1, i2)->sqrt((xs[i1] - xs[i2])^2 + (ys[i1] - ys[i2])^2 + (zs[i1] - zs[i2])^2), n1, n2)
	areasoverlengths = areas ./ lengths
	dirichletnodes = collect(filter(i->isdirichletnode(xs[i], ys[i], zs[i]), 1:length(xs)))
	dirichletheads = map(i->dirichlethead(xs[i], ys[i], zs[i]), dirichletnodes)
	JLD.save(joinpath(meshdir, "mesh.jld"), "xs", xs, "ys", ys, "zs", zs, "neighbors", neighbors, "areasoverlengths", areasoverlengths, "fractureindices", fractureindices, "dirichletnodes", dirichletnodes, "dirichletheads", dirichletheads, "conductivities", conductivities)
end

function setupmesh(meshdir, isdirichletnode, dirichlethead)
	@show meshdir
	badmesh2goodmesh(meshdir)
	goodmesh2greatmesh(meshdir, isdirichletnode, dirichlethead)
end

#@time setupmesh("fourfractures", (x, y, z)->abs(x) == 0.5, (x, y, z)->ifelse(x > 0, 1e6, 2e6))
#@time setupmesh("circuit", (x, y, z)->abs(x) == 1.0, (x, y, z)->ifelse(x > 0, 1e6, 2e6))
#@time setupmesh("25L_network_x2", (x, y, z)->abs(x) > 4.99999, (x, y, z)->ifelse(x > 0, 1e6, 2e6))
#@time setupmesh("homogenous-10m", (x, y, z)->abs(x) > 4.99999, (x, y, z)->ifelse(x > 0, 1e6, 2e6))
@time setupmesh("backbone_x01", (x, y, z)->abs(x) > 7.49999, (x, y, z)->ifelse(x > 0, 1e6, 2e6))
#@time setupmesh("pl_alpha_1.6", (x, y, z)->abs(x) > 499.999, (x, y, z)->ifelse(x > 0, 1e6, 2e6))
