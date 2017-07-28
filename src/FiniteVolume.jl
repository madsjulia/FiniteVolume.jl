module FiniteVolume

import IterativeSolvers
import LinearAdjoints
#import Preconditioners
import PyAMG

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

@LinearAdjoints.assemblesparsematrix (conductivities, sources) u function assembleA(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector)
	I = Int[]
	J = Int[]
	V = Float64[]
	freenode, nodei2freenodei = getfreenodes(length(sources), dirichletnodes)
	for (i, (node1, node2)) in enumerate(neighbors)
		if freenode[node1] && node1 != node2
			v = conductivities[i] * areasoverlengths[i]
			LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], conductivities[i] * areasoverlengths[i])
			if freenode[node2]
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node2], -conductivities[i] * areasoverlengths[i])
			end
		end
	end
	return sparse(I, J, V, sum(freenode), sum(freenode), +)
end

@LinearAdjoints.assemblevector (kx, ky, kz, u_dV, fluxV, gwsink) b function assembleb(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector)
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
		if freenode[node1] && node1 != node2
			if !freenode[node2]
				b[nodei2freenodei[node1]] += conductivities[i] * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]]
			end
		end
	end
	return b
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

end
