import Interpolations

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
	volumes = Array{Float64}(0)
	for i1 = 1:ns[1]
		areadx = xs[2] - xs[1]
		if i1 == 1 || i1 == ns[1]#don't include the area outside the domain
			areadx *= 0.5
		end
		for i2 = 1:ns[2]
			aready = ys[2] - ys[1]
			if i2 == 1 || i2 == ns[2]#don't include the area outside the domain
				aready *= 0.5
			end
			for i3 = 1:ns[3]
				areadz = zs[2] - zs[1]
				if i3 == 1 || i3 == ns[3]#don't include the area outside the domain
					areadz *= 0.5
				end
				push!(volumes, areadx * aready * areadz)
				coords[1, is2k(i1, i2, i3)] = xs[i1]
				coords[2, is2k(i1, i2, i3)] = ys[i2]
				coords[3, is2k(i1, i2, i3)] = zs[i3]
				if i1 < ns[1]
					neighbors[j] = is2k(i1, i2, i3)=>is2k(i1 + 1, i2, i3)
					areasoverlengths[j] = aready * areadz / dx
					j += 1
				end
				if i2 < ns[2]
					neighbors[j] = is2k(i1, i2, i3)=>is2k(i1, i2 + 1, i3)
					areasoverlengths[j] = areadx * areadz / dy
					j += 1
				end
				if i3 < ns[3]
					neighbors[j] = is2k(i1, i2, i3)=>is2k(i1, i2, i3 + 1)
					areasoverlengths[j] = areadx * aready / dz
					j += 1
				end
			end
		end
	end
	return coords, neighbors, areasoverlengths, volumes
end

function makeinterpolant(mins, maxs, ns, h)
	xs = linspace(mins[1], maxs[1], ns[1])
	ys = linspace(mins[2], maxs[2], ns[2])
	zs = linspace(mins[3], maxs[3], ns[3])
	itp = Interpolations.interpolate((zs, ys, xs), reshape(h, length(zs), length(ys), length(xs)), Interpolations.Gridded(Interpolations.Linear()))
	return (x, y, z)->itp[z, y, x]
end
