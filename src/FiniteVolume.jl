module FiniteVolume

import IterativeSolvers
#import Preconditioners
import PyAMG

function mydist(x1, x2)
	return sqrt((x1[1] - x2[1]) ^ 2 + (x1[2] - x2[2]) ^ 2 + (x1[3] - x2[3]) ^ 2)
end

function solvediffusion(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector)
	I = Int[]
	J = Int[]
	V = Float64[]
	nodei2dirichleti = fill(-1, length(sources))
	for i = 1:length(dirichletnodes)
		node = dirichletnodes[i]
		nodei2dirichleti[node] = i
		if sources[node] != 0
			error("There cannot be a source at a Dirichlet node, but node $node is a Dirichlet node where a source is located.")
		end
	end
	freenode = fill(true, length(sources))
	freenode[dirichletnodes] = false
	b = Array{Float64}(sum(freenode))
	nodei2freenodei = fill(-1, length(freenode))
	j = 1
	for i = 1:length(freenode)
		if freenode[i]
			nodei2freenodei[i] = j
			b[j] = sources[i]
			j += 1
		end
	end
	for (i, (node1, node2)) in enumerate(neighbors)
		if freenode[node1] && node1 != node2
			v = conductivities[i] * areasoverlengths[i]
			push!(I, nodei2freenodei[node1])
			push!(J, nodei2freenodei[node1])
			push!(V, v)
			if freenode[node2]
				push!(I, nodei2freenodei[node1])
				push!(J, nodei2freenodei[node2])
				push!(V, -v)
			else
				b[nodei2freenodei[node1]] += v * dirichletheads[nodei2dirichleti[node2]]
			end
		end
	end
	A = sparse(I, J, V, sum(freenode), sum(freenode), +)
	M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	result, ch = IterativeSolvers.gmres(A, b; Pl=M, log=true, maxiter=400, restart=400)
	#result, ch = IterativeSolvers.cg(A, b; Pl=M, log=true, maxiter=400)
	#result = PyAMG.solve(PyAMG.RugeStubenSolver(A), b, accel="cg", tol=1e-4)
	#=
	@time iL, iU = Preconditioners.ilu0(A)
	@time result, ch = IterativeSolvers.gmres(A, b; Pl=iL, Pr=iU, log=true, maxiter=400, restart=400)
	=#
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
