module FiniteVolume

import IterativeSolvers

function mydist(x1, x2)
	return sqrt((x1[1] - x2[1]) ^ 2 + (x1[2] - x2[2]) ^ 2 + (x1[3] - x2[3]) ^ 2)
end

function solvediffusion(volumes::Vector, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector)
	I = Int[]
	J = Int[]
	V = Float64[]
	b = zeros(length(sources))
	@time for i = 1:length(dirichletnodes)
		node = dirichletnodes[i]
		push!(I, node)
		push!(J, node)
		push!(V, 1.0)
		b[node] = dirichletheads[i]
		if sources[node] != 0
			error("There cannot be a source at a Dirichlet node, but node $node is a Dirichlet node where a source is located.")
		end
	end
	freenode = fill(true, length(volumes))
	freenode[dirichletnodes] = false
	@time for (i, (node1, node2)) in enumerate(neighbors)
		if freenode[node1]
			v = conductivities[i] * areasoverlengths[i]
			push!(I, node1)
			push!(J, node1)
			push!(V, v)
			push!(I, node1)
			push!(J, node2)
			push!(V, -v)
			b[node1] = sources[node1]
		end
	end
	@time A = sparse(I, J, V)
	@time result = A \ b
	#result = 1
	#@time result, ch = IterativeSolvers.gmres(A, b; Pl=spdiagm(diag(A)), log=true, maxiter=200, restart=100)
	#@time result, ch = IterativeSolvers.gmres(A, b; log=true, maxiter=100)
	#=
	@show vecnorm(result)
	@show vecnorm(A * result - b)
	@show vecnorm(b)
	@show vecnorm(A * result - b) / vecnorm(b)
	=#
	return A, result, ch
end

end
