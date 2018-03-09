module FiniteVolume

import Interpolations
import IterativeSolvers
import LinearAdjoints
import NearestNeighbors
#import Preconditioners
import PyAMG
import QuadGK

include("grid.jl")
include("transient.jl")
include("transientadjointutils.jl")

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

@LinearAdjoints.assemblesparsematrix (conductivities, sources, dirichletheads) u function assembleA(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i, logtransformconductivity::Bool=false)
	I = Int[]
	J = Int[]
	V = Float64[]
	freenode, nodei2freenodei = getfreenodes(length(sources), dirichletnodes)
	if logtransformconductivity
		for (i, (node1, node2)) in enumerate(neighbors)
			if freenode[node1] && freenode[node2]
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], exp(conductivities[metaindex(i)]) * areasoverlengths[i])
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node2], -exp(conductivities[metaindex(i)]) * areasoverlengths[i])
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node2], exp(conductivities[metaindex(i)]) * areasoverlengths[i])
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node1], -exp(conductivities[metaindex(i)]) * areasoverlengths[i])
			elseif freenode[node1]
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], exp(conductivities[metaindex(i)]) * areasoverlengths[i])
			elseif freenode[node2]
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node2], exp(conductivities[metaindex(i)]) * areasoverlengths[i])
			end
		end
	else
		for (i, (node1, node2)) in enumerate(neighbors)
			if freenode[node1] && freenode[node2]
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], conductivities[metaindex(i)] * areasoverlengths[i])
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node2], -conductivities[metaindex(i)] * areasoverlengths[i])
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node2], conductivities[metaindex(i)] * areasoverlengths[i])
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node1], -conductivities[metaindex(i)] * areasoverlengths[i])
			elseif freenode[node1]
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node1], nodei2freenodei[node1], conductivities[metaindex(i)] * areasoverlengths[i])
			elseif freenode[node2]
				LinearAdjoints.addentry(I, J, V, nodei2freenodei[node2], nodei2freenodei[node2], conductivities[metaindex(i)] * areasoverlengths[i])
			end
		end
	end
	return sparse(I, J, V, sum(freenode), sum(freenode), +)
end

@LinearAdjoints.assemblevector (conductivities, sources, dirichletheads) b function assembleb(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i, logtransformconductivity::Bool=false)
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
	if logtransformconductivity
		for (i, (node1, node2)) in enumerate(neighbors)
			if freenode[node1] && !freenode[node2]
				b[nodei2freenodei[node1]] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]]
			elseif !freenode[node1] && freenode[node2]
				b[nodei2freenodei[node2]] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node1]]
			end
		end
	else
		for (i, (node1, node2)) in enumerate(neighbors)
			if freenode[node1] && !freenode[node2]
				b[nodei2freenodei[node1]] += conductivities[metaindex(i)] * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]]
			elseif !freenode[node1] && freenode[node2]
				b[nodei2freenodei[node2]] += conductivities[metaindex(i)] * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node1]]
			end
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

function solvediffusion(neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector; maxiter=400)
	A = assembleA(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads)
	b = assembleb(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads)
	M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	result, ch = IterativeSolvers.cg(A, b; Pl=M, log=true, maxiter=maxiter)
	#=
	result, ch = IterativeSolvers.gmres(A, b; Pl=M, log=true, maxiter=400, restart=400)
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

function gethycocoords(neighbors, coords)
	hycocoords = Array{Float64}(size(coords, 1), length(neighbors))
	for i = 1:length(neighbors)
		for j = 1:size(coords, 1)
			hycocoords[j, i] = 0.5 * (coords[j, neighbors[i][1]] + coords[j, neighbors[i][2]])
		end
	end
	return hycocoords
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
	I = Array{Int}(2 * length(idxs) * numneighbors)
	J = Array{Int}(2 * length(idxs) * numneighbors)
	V = Array{Float64}(2 * length(idxs) * numneighbors)
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

function simpleintegrate(fs, ts)
	result = similar(fs[1])
	@. result = 0.5 * ((ts[2] - ts[1]) * fs[1] + (ts[end] - ts[end - 1]) * fs[end])
	for i = 2:length(ts) - 1
		@. result += 0.5 * (ts[i + 1] - ts[i - 1]) * fs[i]
	end
	return result
end

function integrateb_pmA_pxlambda(lambdas::Vector{T}, ts_lambda::Vector, u2, tspan, Ss, volumes, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i, logtransformconductivity::Bool=false) where {T <: AbstractArray}
	nodei2dirichleti = getnodei2dirichleti(sources, dirichletnodes)
	freenode, nodei2freenodei = getfreenodes(length(sources), dirichletnodes)
	lambda2 = FiniteVolume.getcontinuoussolution(lambdas, ts_lambda, Val{2})
	result = zeros(length(conductivities) + length(sources) + length(dirichletheads))
	productdict = Dict{Int, Float64}()
	function integrateproduct(i)
		if haskey(productdict, i)
			return productdict[i]
		else
			I = QuadGK.quadgk(t->lambda2[nodei2freenodei[i], t] * u2[i, t], tspan[1], tspan[2])[1]
			productdict[i] = I
			return I
		end
	end
	lambdaintegral = simpleintegrate(lambdas, ts_lambda)
	function getlinearindex(::Type{Val{:conductivities}})
		return 0 + 1
	end
	function getlinearindex(::Type{Val{:conductivities}}, indices...)
		linearindex = 0 + 1
		for i = 1:length(indices)
			offset = 1
			for j = 1:i - 1
				offset *= size(conductivities, j)
			end
			linearindex += (indices[i] - 1) * offset
		end
		return linearindex
	end
	function getlinearindex(::Type{Val{:sources}})
		return (0 + length(conductivities)) + 1
	end
	function getlinearindex(::Type{Val{:sources}}, indices...)
		linearindex = (0 + length(conductivities)) + 1
		for i = 1:length(indices)
			offset = 1
			for j = 1:i - 1
				offset *= size(sources, j)
			end
			linearindex += (indices[i] - 1) * offset
		end
		return linearindex
	end
	function getlinearindex(::Type{Val{:dirichletheads}})
		return ((0 + length(conductivities)) + length(sources)) + 1
	end
	function getlinearindex(::Type{Val{:dirichletheads}}, indices...)
		linearindex = ((0 + length(conductivities)) + length(sources)) + 1
		for i = 1:length(indices)
			offset = 1
			for j = 1:i - 1
				offset *= size(dirichletheads, j)
			end
			linearindex += (indices[i] - 1) * offset
		end
		return linearindex
	end
	b = Array{Float64}(sum(freenode))
	j = 1
	for i = 1:length(freenode)
		if freenode[i]
			#LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:sources}, i), j, 1)
			#result[getlinearindex(Val{:sources}, i)] += lambdaintegrate(j) * Ss * volumes[j]
			result[getlinearindex(Val{:sources}, i)] += lambdaintegral[j] / (Ss * volumes[i])
			j += 1
		end
	end
	#TODO this loop is slow -- make it faster
	if logtransformconductivity
		for (i, (node1, node2)) = enumerate(neighbors)
			if freenode[node1] && !(freenode[node2])
				#LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:conductivities}, metaindex(i)), nodei2freenodei[node1], exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]])
				#LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node2]), nodei2freenodei[node1], exp(conductivities[metaindex(i)]) * areasoverlengths[i])
				#result[getlinearindex(Val{:conductivities}, metaindex(i))] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]] * lambda[nodei2freenodei[node1]] * Ss * volumes[nodei2freenodei[node1]]
				#result[getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node2])] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * lambda[nodei2freenodei[node1]] * Ss * volumes[nodei2freenodei[node1]]
				#result[getlinearindex(Val{:conductivities}, metaindex(i))] += (exp(conductivities[metaindex(i)]) * areasoverlengths[i]) * u[nodei2freenodei[node1]] * lambda[nodei2freenodei[node1]] * Ss * volumes[nodei2freenodei[node1]]
				result[getlinearindex(Val{:conductivities}, metaindex(i))] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]] * lambdaintegral[nodei2freenodei[node1]] / (Ss * volumes[nodei2freenodei[node1]])
				result[getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node2])] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * lambdaintegral[nodei2freenodei[node1]] / (Ss * volumes[nodei2freenodei[node1]])
				result[getlinearindex(Val{:conductivities}, metaindex(i))] += (exp(conductivities[metaindex(i)]) * areasoverlengths[i]) * integrateproduct(node1) / (Ss * volumes[nodei2freenodei[node1]])
			elseif !(freenode[node1]) && freenode[node2]
				#LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:conductivities}, metaindex(i)), nodei2freenodei[node2], exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node1]])
				#LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node1]), nodei2freenodei[node2], exp(conductivities[metaindex(i)]) * areasoverlengths[i])
				#result[getlinearindex(Val{:conductivities}, metaindex(i))] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node1]] * lambda[nodei2freenodei[node2]] * Ss * volumes[nodei2freenodei[node2]]
				#result[getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node1])] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * lambda[nodei2freenodei[node2]] * Ss * volumes[nodei2freenodei[node2]]
				#result[getlinearindex(Val{:conductivities}, metaindex(i))] += (exp(conductivities[metaindex(i)]) * areasoverlengths[i]) * u[nodei2freenodei[node2]] * lambda[nodei2freenodei[node2]] * Ss * volumes[nodei2freenodei[node2]]
				result[getlinearindex(Val{:conductivities}, metaindex(i))] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * dirichletheads[nodei2dirichleti[node1]] * lambdaintegral[nodei2freenodei[node2]] / (Ss * volumes[nodei2freenodei[node2]])
				result[getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node1])] += exp(conductivities[metaindex(i)]) * areasoverlengths[i] * lambdaintegral[nodei2freenodei[node2]] / (Ss * volumes[nodei2freenodei[node2]])
				result[getlinearindex(Val{:conductivities}, metaindex(i))] += (exp(conductivities[metaindex(i)]) * areasoverlengths[i]) * integrateproduct(node2) / (Ss * volumes[nodei2freenodei[node2]])
			end
		end
	else
		error("not supported")
		#=
		for (i, (node1, node2)) = enumerate(neighbors)
			if freenode[node1] && !(freenode[node2])
				LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:conductivities}, metaindex(i)), nodei2freenodei[node1], areasoverlengths[i] * dirichletheads[nodei2dirichleti[node2]])
				LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node2]), nodei2freenodei[node1], conductivities[metaindex(i)] * areasoverlengths[i])
			elseif !(freenode[node1]) && freenode[node2]
				LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:conductivities}, metaindex(i)), nodei2freenodei[node2], areasoverlengths[i] * dirichletheads[nodei2dirichleti[node1]])
				LinearAdjoints.addentry(___la___I, ___la___J, ___la___V, getlinearindex(Val{:dirichletheads}, nodei2dirichleti[node1]), nodei2freenodei[node2], conductivities[metaindex(i)] * areasoverlengths[i])
			end
		end
		=#
	end
	return result
end

end
