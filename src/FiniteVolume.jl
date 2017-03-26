module FiniteVolume

function mydist(x1, x2)
	return sqrt((x1[1] - x2[1]) ^ 2 + (x1[2] - x2[2]) ^ 2 + (x1[3] - x2[3]) ^ 2)
end

function solvediffusion(xs, neighbors::Array{Array{Int, 1}, 1}, areas, conductivities, sources, dirichletnodes::Array{Int, 1}, dirichletheads)
	I = Int[]
	J = Int[]
	V = Float64[]
	b = zeros(size(xs, 2))
	for i = 1:length(dirichletnodes)
		node = dirichletnodes[i]
		push!(I, node)
		push!(J, node)
		push!(V, 1.0)
		b[node] = dirichletheads[i]
		if sources[node] != 0
			error("There is a source at a Dirichlet node")
		end
	end
	freenodes = setdiff(1:size(xs, 2), dirichletnodes)
	for node1 in freenodes
		for (i, node2) in enumerate(neighbors[node1])
			d = norm(xs[:, node1] - xs[:, node2])
			v = conductivities[node1][i] * areas[node1][i]
			push!(I, node1)
			push!(J, node1)
			push!(V, v)
			push!(I, node1)
			push!(J, node2)
			push!(V, -v)
			b[node1] = sources[node1]
		end
	end
	A = sparse(I, J, V)
	return A \ b
end

end
