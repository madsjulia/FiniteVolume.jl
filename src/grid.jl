function getnearestgridpoint(x, coords)
	bestind = 1
	bestdist = LinearAlgebra.norm(x - coords[:, 1])
	for i = 1:size(coords, 2)
		thisdist = LinearAlgebra.norm(x - coords[:, i])
		if thisdist < bestdist
			bestind = i
			bestdist = thisdist
		end
	end
	return bestind
end

function nodehycos2neighborhycos(neighbors, nodehycos, logtransformhyco=false)
	n1 = size(nodehycos, 1)
	n2 = size(nodehycos, 2)
	n3 = size(nodehycos, 3)
	function multiindex(k)
		i3 = mod(k - 1, n3) + 1
		i2 = mod(div(k - i3, n3), n2) + 1
		i1 = div(k - i3 - (i2 - 1) * n3, n3 * n2) + 1
		return i1, i2, i3
	end
	neighborhycos = Array{Float64}(length(neighbors))
	for i = 1:length(neighborhycos)
		if logtransformhyco
			neighborhycos[i] = 0.5 * (nodehycos[multiindex(neighbors[i][1])...] + nodehycos[multiindex(neighbors[i][2])...])
		else
			neighborhycos[i] = sqrt(nodehycos[multiindex(neighbors[i][1])...] * nodehycos[multiindex(neighbors[i][2])...])
		end
	end
	return neighborhycos
end

function neighborhycos2nodehycos(neighbors, neighborhycos, ns)
	function multiindex(k)
		i3 = mod(k - 1, ns[3]) + 1
		i2 = mod(div(k - i3, ns[3]), ns[2]) + 1
		i1 = div(k - i3 - (i2 - 1) * ns[3], ns[3] * ns[2]) + 1
		return i1, i2, i3
	end
	nodehycos = ones(Float64, ns...)
	neighborcount = zeros(Int, ns...)
	for i = 1:length(neighborhycos)
		#neighborhycos[i] = sqrt(nodehycos[multiindex(neighbors[i][1])...] * nodehycos[multiindex(neighbors[i][2])...])
		nodehycos[multiindex(neighbors[i][1])...] += neighborhycos[i]
		neighborcount[multiindex(neighbors[i][1])...] += 1
		nodehycos[multiindex(neighbors[i][2])...] += neighborhycos[i]
		neighborcount[multiindex(neighbors[i][2])...] += 1
	end
	@. nodehycos /= neighborcount
	return nodehycos
end


function regulargrid(mins, maxs, ns)
	@assert length(mins) == length(maxs)
	@assert length(mins) == length(ns)
	length(mins) == 3 || error("only 3 dimensions supported")
	linearindex = (i1, i2, i3)->i3 + ns[3] * (i2 - 1) + ns[3] * ns[2] * (i1 - 1)
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
				coords[1, linearindex(i1, i2, i3)] = xs[i1]
				coords[2, linearindex(i1, i2, i3)] = ys[i2]
				coords[3, linearindex(i1, i2, i3)] = zs[i3]
				if i1 < ns[1]
					neighbors[j] = linearindex(i1, i2, i3)=>linearindex(i1 + 1, i2, i3)
					areasoverlengths[j] = aready * areadz / dx
					j += 1
				end
				if i2 < ns[2]
					neighbors[j] = linearindex(i1, i2, i3)=>linearindex(i1, i2 + 1, i3)
					areasoverlengths[j] = areadx * areadz / dy
					j += 1
				end
				if i3 < ns[3]
					neighbors[j] = linearindex(i1, i2, i3)=>linearindex(i1, i2, i3 + 1)
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
