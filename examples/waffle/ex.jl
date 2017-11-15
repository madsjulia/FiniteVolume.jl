using Base.Test
import FiniteVolume
import JLD
import LinearAdjoints
import Optim
import PyPlot
import ReusableFunctions

doplot = false

@time neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads = JLD.load("waffledata.jld", "neighbors", "areasoverlengths", "hycos", "sources", "dirichletnodes", "dirichletheads")

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
const regularization = 1e0
if !isdefined(:regmat)
	const regmat = FiniteVolume.hycoregularizationmatrix(neighbors, length(sources))
else
	println("skipping regmat")
end
function objfunc(u, neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
	of = 0.0
	head, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources, dirichletnodes, dirichletheads)
	for i = 1:length(obsnodes)
		of += (obsvalues[i] - head[obsnodes[i]])^2
	end
	of += regularization * norm(regmat * hycos)^2
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
	of_p[1:length(hycos)] = (2 * regularization) * At_mul_B(regmat, (regmat * hycos))
	return of_p
end
function setupsolver(A)
	#M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	rss = PyAMG.RugeStubenSolver(A)
	function solver(b, transpose=false)
		#matrix is symmetric, so we don't need to deal with transpose
		result = PyAMG.solve(rss, b, accel="cg", tol=sqrt(eps(Float64)))
		#result, ch = IterativeSolvers.gmres(A, b; Pl=M, log=true, maxiter=100, restart=100, tol=sqrt(eps(Float64)))
		if !ch.isconverged
			warn("may not be converged")
		end
		return result
	end
end
@LinearAdjoints.adjoint adjoint FiniteVolume.assembleA FiniteVolume.assembleb objfunc objfunc_x objfunc_p setupsolver
@time u, of, gradient = adjoint(neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
adjointhead, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources, dirichletnodes, dirichletheads)

r3adjoint = ReusableFunctions.maker3function(adjoint, "restarts")
#compare the gradient to a finite difference gradient
is = [rand(1:length(hycos), 2); length(hycos) + rand(1:length(sources), 2); length(hycos) + length(sources) + rand(1:length(dirichletheads), 2)]
function objfunc(x)
	global hycos
	global sources
	global dirichletheads
	thesehycos = x[1:length(hycos)]
	thesesources = x[length(hycos) + 1:length(hycos) + length(sources)]
	thesedirichletheads = x[length(hycos) + length(sources) + 1:length(hycos) + length(sources) + length(dirichletheads)]
	u, of, gradient = r3adjoint(neighbors, areasoverlengths, thesehycos, thesesources, dirichletnodes, thesedirichletheads)
	@show of
	return of
end
x0 = [hycos; sources; dirichletheads]
#=
smallgrad = Array{Float64}(length(is))
deltax = 1e-8
for i = 1:length(is)
	@show i
	x = copy(x0)
	x[is[i]] += deltax
	ofnew = objfunc(x)
	smallgrad[i] = (ofnew - of) / deltax
end
@show smallgrad
@show gradient[is]
=#

function gradient!(storage, x)
	global hycos
	global sources
	global dirichletheads
	thesehycos = x[1:length(hycos)]
	thesesources = x[length(hycos) + 1:length(hycos) + length(sources)]
	thesedirichletheads = x[length(hycos) + length(sources) + 1:length(hycos) + length(sources) + length(dirichletheads)]
	u, of, gradient = r3adjoint(neighbors, areasoverlengths, thesehycos, thesesources, dirichletnodes, thesedirichletheads)
	@show of
	copy!(storage, gradient)
end
@time opt = Optim.optimize(objfunc, gradient!, x0, Optim.LBFGS(), Optim.Options(iterations=10, show_trace=true))
x = opt.minimizer
opthycos = x[1:length(hycos)]
optsources = x[length(hycos) + 1:length(hycos) + length(sources)]
optdirichletheads = x[length(hycos) + length(sources) + 1:length(hycos) + length(sources) + length(dirichletheads)]
u, of, gradient = r3adjoint(neighbors, areasoverlengths, opthycos, optsources, dirichletnodes, optdirichletheads)
of = 0.0
opthead, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources, dirichletnodes, dirichletheads)
@show norm(obsvalues - opthead[obsnodes])^2
@show norm(obsvalues - adjointhead[obsnodes])^2

nothing
