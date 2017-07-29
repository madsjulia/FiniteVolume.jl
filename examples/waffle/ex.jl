using Base.Test
import FiniteVolume
import JLD
import LinearAdjoints
import PyPlot

doplot = false

@time neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads = JLD.load("waffledata.jld", "neighbors", "areasoverlengths", "hycos", "sources", "dirichletnodes", "dirichletheads")

#=
#do the uniform hyco, no recharge case
@time head, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, ones(length(areasoverlengths)), zeros(length(sources)), dirichletnodes, dirichletheads)
fehmhead = readdlm("data/w01.00007_sca_node.avs.hom_norecharge"; skipstart=2)[:, 2]
@show norm(A * head[freenode] - b)
@show norm(A * fehmhead[freenode] - b)
@test norm(A * fehmhead[freenode] - b) / norm(b) < 1e-3
@show norm(b)
@show norm(head - fehmhead)
@show mean(head - fehmhead)
@show mean(abs.(head - fehmhead))
if doplot
	fig, ax = PyPlot.subplots()
	ax[:plot](log10.(ch.data[:resnorm]))
	display(fig); println()
	PyPlot.close(fig)
end

#do the heterogeneous hyco, no recharge case
@time hethead, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, hycos, zeros(length(sources)), dirichletnodes, dirichletheads)
fehmhethead = readdlm("data/w01.00007_sca_node.avs.het_norecharge"; skipstart=2)[:, 2]
@show norm(A * hethead[freenode] - b)
@show norm(A * fehmhethead[freenode] - b)
@test norm(A * fehmhethead[freenode] - b) / norm(b) < 1e-3
@show norm(b)
@show norm(hethead - fehmhethead)
@show mean(hethead - fehmhethead)
@show mean(abs.(hethead - fehmhethead))
if doplot
	fig, ax = PyPlot.subplots()
	ax[:plot](log10.(ch.data[:resnorm]))
	display(fig); println()
	PyPlot.close(fig)
end
=#

#do the heterogeneous hyco, recharge case
@time fullhead, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
fehmfullhead = readdlm("data/w01.00007_sca_node.avs.het_recharge"; skipstart=2)[:, 2]
@show norm(A * fullhead[freenode] - b)
@show norm(A * fehmfullhead[freenode] - b)
@test norm(A * fehmfullhead[freenode] - b) / norm(b) < 1e-3
@show norm(b)
@show norm(fullhead - fehmfullhead)
@show mean(fullhead - fehmfullhead)
@show mean(abs.(fullhead - fehmfullhead))
if doplot
	fig, ax = PyPlot.subplots()
	ax[:plot](log10.(ch.data[:resnorm]))
	display(fig); println()
	PyPlot.close(fig)
end

srand(0)
const obsnodes = rand(1:length(sources), 30)
const obsvalues = fullhead[obsnodes] + 0.1 * randn(length(obsnodes))
const regularization = 1e-1
function objfunc(u, neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
	of = 0.0
	head, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources, dirichletnodes, dirichletheads)
	for i = 1:length(obsnodes)
		of += (obsvalues[i] - head[obsnodes[i]])^2
	end
	@show of
	neighbordict = Dict{Int, Set{Int}}()
	for i = 1:length(sources)
		neighbordict[i] = Set{Int}()
	end
	for i = 1:length(neighbors)
		node1, node2 = neighbors[i]
		if node1 != node2
			push!(neighbordict[node1], node2)
			push!(neighbordict[node2], node1)
		end
	end
	hycodict = Dict(zip(neighbors, hycos))
	for i1 = 1:length(sources)
		for i2 in neighbordict[i1]
			for i3 in neighbordict[i1]
				if i2 < i3
					of += regularization * (hycodict[i1=>i2] - hycodict[i1=>i3])^2
				end
			end
		end
	end
	@show of
	return of
end
function objfunc_x(u, neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
	of_x = zeros(length(u))
	head, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources, dirichletnodes, dirichletheads)
	for i = 1:length(obsnodes)
		of_x[nodei2freenodei[obsnodes[i]]] -= 2 * (obsvalues[i] - head[obsnodes[i]])
	end
	return of_x
end
function objfunc_p(u, neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
	of_p = zeros(sum(map(length, Any[hycos, sources, dirichletheads])))
	neighbordict = Dict{Int, Set{Int}}()
	for i = 1:length(sources)
		neighbordict[i] = Set{Int}()
	end
	for i = 1:length(neighbors)
		node1, node2 = neighbors[i]
		if node1 != node2
			push!(neighbordict[node1], node2)
			push!(neighbordict[node2], node1)
		end
	end
	hycoindices = Dict(zip(neighbors, 1:length(hycos)))
	for i1 = 1:length(sources)
		for i2 in neighbordict[i1]
			for i3 in neighbordict[i1]
				if i2 < i3
					of_p[hycoindices[i1=>i2]] += 2 * regularization * (hycos[hycoindices[i1=>i2]] - hycos[hycoindices[i1=>i3]])
					of_p[hycoindices[i1=>i3]] -= 2 * regularization * (hycos[hycoindices[i1=>i2]] - hycos[hycoindices[i1=>i3]])
				end
			end
		end
	end
	return of_p
end
function setupsolver(A)
	M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	function solver(b, transpose=false)
		#matrix is symmetric, so we don't need to deal with transpose
		#result = PyAMG.solve(PyAMG.RugeStubenSolver(A), b, accel="cg", tol=sqrt(eps(Float64)))
		#result = PyAMG.solve(PyAMG.RugeStubenSolver(A), b, accel="cg", tol=1e-10)
		result, ch = IterativeSolvers.gmres(A, b; Pl=M, log=true, maxiter=400, restart=400, tol=1e-14)
		if !ch.isconverged
			warn("may not be converged")
		end
		return result
	end
end
@LinearAdjoints.adjoint adjoint FiniteVolume.assembleA FiniteVolume.assembleb objfunc objfunc_x objfunc_p setupsolver
@time u, of, gradient = adjoint(neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
adjointhead, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources, dirichletnodes, dirichletheads)

is = [rand(1:length(hycos), 2); length(hycos) + rand(1:length(sources), 2); length(hycos) + length(sources) + rand(1:length(dirichletheads), 2)]
function objfunc(x)
	global hycos
	global sources
	global dirichletheads
	thesehycos = x[1:length(hycos)]
	thesesources = x[length(hycos) + 1:length(hycos) + length(sources)]
	thesedirichletheads = x[length(hycos) + length(sources) + 1:length(hycos) + length(sources) + length(dirichletheads)]
	@time u, of, gradient = adjoint(neighbors, areasoverlengths, thesehycos, thesesources, dirichletnodes, thesedirichletheads)
	return of
end
smallgrad = Array{Float64}(length(is))
x0 = [hycos; sources; dirichletheads]
deltax = 1e-8
for i = 1:length(is)
	@show i
	x = copy(x0)
	x[is[i]] += deltax
	ofnew = objfunc(x)
	smallgrad[i] = (ofnew - of) / deltax
end
nothing
