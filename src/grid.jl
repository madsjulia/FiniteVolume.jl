function regulargrid(mins, maxs, ns)
	@assert length(mins) == length(maxs)
	@assert length(mins) == length(ns)
	length(mins) == 3 || error("only 3 dimensions supported")
	is2k = (i1, i2, i3)->i3 + ns[3] * (i2 - 1) + ns[3] * ns[2] * (i1 - 1)
	coords = Array{Float64}(3, prod(ns))
	xs = linspace(mins[1], maxs[1], ns[1])
	ys = linspace(mins[2], maxs[2], ns[2])
	zs = linspace(mins[3], maxs[3], ns[3])
	dx = xs[2] - xs[1]
	dy = ys[2] - ys[1]
	dz = zs[2] - zs[1]
	j = 1
	neighbors = Array{Pair{Int, Int}}(3 * prod(ns) - ns[1] * ns[2] - ns[1] * ns[3] - ns[2] * ns[3])
	areasoverlengths = Array{Float64}(3 * prod(ns) - ns[1] * ns[2] - ns[1] * ns[3] - ns[2] * ns[3])
	for i1 = 1:ns[1]
		for i2 = 1:ns[2]
			for i3 = 1:ns[3]
				coords[1, is2k(i1, i2, i3)] = xs[i1]
				coords[2, is2k(i1, i2, i3)] = ys[i2]
				coords[3, is2k(i1, i2, i3)] = zs[i3]
				if i1 < ns[1]
					neighbors[j] = is2k(i1, i2, i3)=>is2k(i1 + 1, i2, i3)
					areasoverlengths[j] = dy * dz / dx
					j += 1
				end
				if i2 < ns[2]
					neighbors[j] = is2k(i1, i2, i3)=>is2k(i1, i2 + 1, i3)
					areasoverlengths[j] = dx * dz / dy
					j += 1
				end
				if i3 < ns[3]
					neighbors[j] = is2k(i1, i2, i3)=>is2k(i1, i2, i3 + 1)
					areasoverlengths[j] = dx * dy / dz
					j += 1
				end
			end
		end
	end
	return coords, neighbors, areasoverlengths
end
