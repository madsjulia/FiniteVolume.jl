import JLD

function badmesh2goodmesh(meshdir="fourfractures/")
	lines = readlines(joinpath(meshdir, "full_mesh_vol_area.uge"))
	numcells = parse(Int, split(lines[1])[2])
	numconnections = parse(Int, split(lines[numcells + 2])[2])
	cellfile = open(joinpath(meshdir, "cells.dlm"), "w")
	for i = 2:numcells + 1
		write(cellfile, lines[i])
		write(cellfile, "\n")
	end
	close(cellfile)
	connectionfile = open(joinpath(meshdir, "connections.dlm"), "w")
	for i = numcells + 3:numcells + 3 + numconnections - 1
		write(connectionfile, lines[i])
		write(connectionfile, "\n")
	end
	close(connectionfile)
end

function goodmesh2greatmesh(meshdir, isdirichletnode, dirichlethead)
	celldata::Array{Float64, 2} = readdlm(joinpath(meshdir, "cells.dlm"))
	connectiondata::Array{Float64, 2} = readdlm(joinpath(meshdir, "connections.dlm"))
	fractureindices::Array{Int, 1} = readdlm(joinpath(meshdir, "materialid.dat"); skipstart=3)[:]
	xs = celldata[:, 2]
	ys = celldata[:, 3]
	zs = celldata[:, 4]
	n1 = map(Int, connectiondata[:, 1])
	n2 = map(Int, connectiondata[:, 2])
	neighbors = map(Pair, n1, n2)
	areas = connectiondata[:, end]
	lengths = map((i1, i2)->sqrt((xs[i1] - xs[i2])^2 + (ys[i1] - ys[i2])^2 + (zs[i1] - zs[i2])^2), n1, n2)
	areasoverlengths = areas ./ lengths
	dirichletnodes = collect(filter(i->isdirichletnode(xs[i], ys[i], zs[i]), 1:length(xs)))
	dirichletheads = map(i->dirichlethead(xs[i], ys[i], zs[i]), dirichletnodes)
	JLD.save(joinpath(meshdir, "mesh.jld"), "xs", xs, "ys", ys, "zs", zs, "neighbors", neighbors, "areasoverlengths", areasoverlengths, "fractureindices", fractureindices, "dirichletnodes", dirichletnodes, "dirichletheads", dirichletheads)
end

function setupmesh(meshdir, isdirichletnode, dirichlethead)
	badmesh2goodmesh(meshdir)
	goodmesh2greatmesh(meshdir, isdirichletnode, dirichlethead)
end

setupmesh("fourfractures", (x, y, z)->abs(x) == 0.5, (x, y, z)->ifelse(x > 0, 1e6, 2e6))
