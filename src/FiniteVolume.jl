module FiniteVolume

import IterativeSolvers
import LinearAdjoints
#import Preconditioners
import PyAMG
import NearestNeighbors

function mydist(x1, x2)
	return sqrt((x1[1] - x2[1]) ^ 2 + (x1[2] - x2[2]) ^ 2 + (x1[3] - x2[3]) ^ 2)
end

function getnodei2dirichleti(sources, dirichletnodes)
	nodei2dirichleti = fill(-1, length(sources))
	for i = 1:length(dirichletnodes)
		node = dirichletnodes[i]
		nodei2dirichleti[node] = i
		if sources[node] != 0
			error("There cannot be a source at a Dirichlet node, but node $node is a Dirichlet node where a source is located.")
		end
	end
	return nodei2dirichleti
end

function getfreenodes(n, dirichletnodes)
	freenode = fill(true, n)
	freenode[dirichletnodes] = false
	nodei2freenodei = fill(-1, length(freenode))
	j = 1
	for i = 1:length(freenode)
		if freenode[i]
			nodei2freenodei[i] = j
			j += 1
		end
	end
	return freenode, nodei2freenodei
end

function sourceregularizationmatrix(neighbors, areasoverlengths, dirichletnodes, numnodes)
	I = Int[]
	J = Int[]
	V = Float64[]
	freenode, nodei2freenodei = getfreenodes(numnodes, dirichletnodes)
	for (i, (node1, node2)) in enumerate(neighbors)
		if freenode[node1] && freenode[node2]
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], areasoverlengths[i])
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node2], -areasoverlengths[i])
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node2], areasoverlengths[i])
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node1], -areasoverlengths[i])
		end
	end
	return sparse(I, J, V, numnodes, numnodes)
end

@LinearAdjoints.assemblesparsematrix (conductivities, sources, dirichletheads) u function assembleA(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector)
	I = Int[]
	J = Int[]
	V = Float64[]
	freenode, nodei2freenodei = getfreenodes(length(sources), dirichletnodes)
	for (i, (node1, node2)) in enumerate(neighbors)
		if freenode[node1] && freenode[node2]
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], conductivities[i] * areasoverlengths[i])
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node2], -conductivities[i] * areasoverlengths[i])
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node2], conductivities[i] * areasoverlengths[i])
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node1], -conductivities[i] * areasoverlengths[i])
		elseif freenode[node1]
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], conductivities[i] * areasoverlengths[i])
		elseif freenode[node2]
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node2], conductivities[i] * areasoverlengths[i])
		end
	end
	return sparse(I, J, V, sum(freenode), sum(freenode), +)
end

@LinearAdjoints.assemblevector (conductivities, sources, dirichletheads) b function assembleb(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector)
	nodei2dirichleti = getnodei2dirichleti(sources, dirichletnodes)
	freenode, nodei2freenodei = getfreenodes(length(sources), dirichletnodes)
	b = Array{Float64}(sum(freenode))
	j = 1
	for i = 1:length(freenode)
		if freenode[i]
			b[j] = sources[i]
			j += 1
		end
	end
	for (i, (node1, node2)) in enumerate(neighbors)
		if freenode[node1] && !freenode[node2]
			b[nodei2freenodei[node1]] += conductivities[i] * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]]
		elseif !freenode[node1] && freenode[node2]
			b[nodei2freenodei[node2]] += conductivities[i] * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node1]]
		end
	end
	return b
end

function freenodes2nodes(result, sources, dirichletnodes, dirichletheads)
	nodei2dirichleti = getnodei2dirichleti(sources, dirichletnodes)
	freenode, nodei2freenodei = getfreenodes(length(sources), dirichletnodes)
	head = Array{Float64}(length(sources))
	freenodessofar = 0
	for i = 1:length(sources)
		if freenode[i]
			freenodessofar += 1
			head[i] = result[freenodessofar]
		else
			head[i] = dirichletheads[nodei2dirichleti[i]]
		end
	end
	return head, freenode, nodei2freenodei
end

function solvediffusion(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector)
	A = assembleA(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads)
	b = assembleb(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads)
	M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	result, ch = IterativeSolvers.gmres(A, b; Pl=M, log=true, maxiter=400, restart=400)
	#=
	result, ch = IterativeSolvers.cg(A, b; Pl=M, log=true, maxiter=400)
	@time result2 = PyAMG.solve(PyAMG.RugeStubenSolver(A), b, accel="cg", tol=sqrt(eps(Float64)))
	@time result3 = PyAMG.solve(PyAMG.RugeStubenSolver(A, accel="cg"), b, tol=sqrt(eps(Float64)))
	@time result4 = PyAMG.solve(PyAMG.SmoothedAggregationSolver(A, accel="cg"), b, tol=sqrt(eps(Float64)))
	@show norm(A * result - b)
	@show norm(A * result2 - b)
	@show norm(A * result3 - b)
	@show norm(A * result4 - b)
	=#
	#=
	@time iL, iU = Preconditioners.ilu0(A)
	@time result, ch = IterativeSolvers.gmres(A, b; Pl=iL, Pr=iU, log=true, maxiter=400, restart=400)
	=#
	head, freenode, nodei2freenodei = freenodes2nodes(result, sources, dirichletnodes, dirichletheads)
	return head, ch, A, b, freenode
end

function fehmhyco2fvhyco(xs, ys, zs, kxs, kys, kzs, neighbors)
	ks = Array{eltype(kxs)}(length(neighbors))
	for i = 1:length(neighbors)
		node1, node2 = neighbors[i]
		dx = xs[node1] - xs[node2]
		dy = ys[node1] - ys[node2]
		dz = zs[node1] - zs[node2]
		dist = sqrt(dx^2 + dy^2 + dz^2)
		k1 = (abs(kxs[node1] * dx) + abs(kys[node1] * dy) + abs(kzs[node1] * dz)) / dist
		k2 = (abs(kxs[node2] * dx) + abs(kys[node2] * dy) + abs(kzs[node2] * dz)) / dist
		ks[i] = sqrt(k1 * k2)
	end
	return ks
end

function hycoregularizationmatrix(neighbors, numnodes)
	I = Int[]
	J = Int[]
	V = Float64[]
	neighbordict = Dict{Int, Set{Int}}()
	for i = 1:numnodes
		neighbordict[i] = Set{Int}()
	end
	for i = 1:length(neighbors)
		node1, node2 = neighbors[i]
		if node1 != node2
			push!(neighbordict[node1], node2)
			push!(neighbordict[node2], node1)
		end
	end
	hycoindices = Dict(zip(neighbors, 1:length(neighbors)))
	for i1 = 1:numnodes
		for i2 in neighbordict[i1]
			if i1 < i2
				for i3 in neighbordict[i1]
					if i2 != i3 && i1 < i3
						push!(I, hycoindices[i1=>i2])
						push!(J, hycoindices[i1=>i2])
						push!(V, 1.0)
						push!(I, hycoindices[i1=>i2])
						push!(J, hycoindices[i1=>i3])
						push!(V, -1.0)
					end
				end
			end
		end
	end
	return sparse(I, J, V, length(neighbors), length(neighbors), +)
end

function knnregularization(coords, numneighbors, weightfun=h->1 / h)
	I, J, V = innerknnregularization(coords, numneighbors, weightfun)
	return sparse(I, J, V, size(coords, 2) * numneighbors, size(coords, 2))
end

function innerknnregularization(coords, numneighbors, weightfun=h->1 / h)
	kdtree = NearestNeighbors.KDTree(coords)
	idxs, dists = NearestNeighbors.knn(kdtree, coords, numneighbors + 1, true)
	return assembleknnregularization(idxs, dists, weightfun)
end

function assembleknnregularization(idxs, dists, weightfun)
	numneighbors = length(idxs[1]) - 1
	I = Array(Int, 2 * length(idxs) * numneighbors)
	J = Array(Int, 2 * length(idxs) * numneighbors)
	V = Array(Float64, 2 * length(idxs) * numneighbors)
	k = 1
	eqnum = 1
	for i = 1:length(idxs)
		for j = 2:numneighbors + 1#start at 2 to exclude the point itself from the "neighbors"
			v = weightfun(dists[i][j])
			I[k] = eqnum
			J[k] = i
			V[k] = v
			k += 1
			I[k] = eqnum
			J[k] = idxs[i][j]
			V[k] = -v
			k += 1
			eqnum += 1
		end
	end
	return I, J, V
end

end
