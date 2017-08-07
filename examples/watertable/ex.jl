import FiniteVolume
import JLD
import LinearAdjoints
import Optim
import PyPlot
import ReusableFunctions
import PyCall
@PyCall.pyimport scipy.interpolate as interp
import PyPlot

@time neighbors, areasoverlengths, dirichletnodes, rechargenodes, xs, ys, zs, topnodes, hycoxs, hycoys, hycozs, tophyconodes = JLD.load("model.jld", "neighbors", "areasoverlengths", "dirichletnodes", "rechargenodes", "xs", "ys", "zs", "topnodes", "hycoxs", "hycoys", "hycozs", "tophyconodes")
const numnodes = maximum(map(maximum, neighbors))

srand(0)
const obsnodes = JLD.load("observations.jld", "obsnodes")
const obsvalues = JLD.load("observations.jld", "obsvalues") * 0.3048#convert feet to meters
const obsxs = JLD.load("observations.jld", "obsxs")
const obsys = JLD.load("observations.jld", "obsys")
const obsnames = JLD.load("observations.jld", "obsnames")
const hycoregularization = 1e-3
const sourceregularization = 1e3
if !isdefined(:hycoregmat)
	const hycoregmat = FiniteVolume.hycoregularizationmatrix(neighbors, numnodes)
	const sourceregmat = FiniteVolume.sourceregularizationmatrix(neighbors, areasoverlengths, dirichletnodes, numnodes)
else
	println("skipping regmats")
end
function objfunc(u, neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
	of = 0.0
	head, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources, dirichletnodes, dirichletheads)
	for i = 1:length(obsnodes)
		of += (obsvalues[i] - head[obsnodes[i]])^2
	end
	hycoregof = hycoregularization * norm(hycoregmat * log.(hycos))^2
	sourceregof = sourceregularization * norm(sourceregmat * sources)^2
	@show of
	@show hycoregof
	@show sourceregof
	of += hycoregof + sourceregof
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
	of_p[1:length(hycos)] = (2 * hycoregularization) * At_mul_B(hycoregmat, (hycoregmat * log.(hycos))) ./ hycos
	of_p[length(hycos) + 1:length(hycos) + length(sources)] = (2 * sourceregularization) * At_mul_B(sourceregmat, (sourceregmat * sources))
	return of_p
end
function setupsolver(A)
	#M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	rss = PyAMG.RugeStubenSolver(A)
	function solver(b, transpose=false)
		#matrix is symmetric, so we don't need to deal with transpose
		result = PyAMG.solve(rss, b, accel="cg", tol=sqrt(eps(Float64)))
		#=
		result, ch = IterativeSolvers.gmres(A, b; Pl=M, log=true, maxiter=100, restart=100, tol=sqrt(eps(Float64)))
		if !ch.isconverged
			warn("may not be converged")
		end
		=#
		return result
	end
end

function atantransform(unboundedparams, lowerbounds, upperbounds)
	return lowerbounds + (upperbounds - lowerbounds) .* (atan.(unboundedparams) / pi + 0.5)
end

function tantransform(boundedparams, lowerbounds, upperbounds)
	return tan.(pi * ((boundedparams - lowerbounds) ./ (upperbounds - lowerbounds) - 0.5))
end

function atantransformgradient!(gradient, unboundedparams, lowerbounds, upperbounds)
	for i = 1:length(gradient)
		gradient[i] *= (upperbounds[i] - lowerbounds[i]) ./ ((1 + unboundedparams[i]^2) * pi)
	end
end

hycos0 = 1e-5 * ones(length(neighbors))#m/s
sources0 = zeros(numnodes)#m^3/s
dirichletheads0 = JLD.load("observations.jld", "dirichletheads0") * 0.3048#convert feet to meters
x0 = [log.(hycos0); zeros(length(rechargenodes)); dirichletheads0]
const lowerbounds = [log.(fill(1e-6, length(hycos0))); fill(-1e-6, length(rechargenodes)); dirichletheads0 - 50]
const upperbounds = [log.(fill(1e-4, length(hycos0))); fill(1e-6, length(rechargenodes)); dirichletheads0 + 50]
x0unbounded = tantransform(x0, lowerbounds, upperbounds)
@show norm(x0 - atantransform(x0unbounded, lowerbounds, upperbounds))

if !isdefined(:opt)
	@LinearAdjoints.adjoint adjoint FiniteVolume.assembleA FiniteVolume.assembleb objfunc objfunc_x objfunc_p setupsolver
	@time u, of, gradient = adjoint(neighbors, areasoverlengths, hycos0, sources0, dirichletnodes, dirichletheads0)
	adjointhead, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, sources0, dirichletnodes, dirichletheads0)

	r3adjoint = ReusableFunctions.maker3function(adjoint, "restarts")
	#r3adjoint = adjoint
	#compare the gradient to a finite difference gradient
	function objfunc(xunbounded)
		global neighbors
		global rechargenodes
		global dirichletnodes
		global numnodes
		x = atantransform(xunbounded, lowerbounds, upperbounds)
		thesehycos = exp.(x[1:length(neighbors)])
		thesesources = zeros(numnodes)
		for (i, node) in enumerate(rechargenodes)
			thesesources[node] = x[length(neighbors) + i]
		end
		thesedirichletheads = x[length(neighbors) + length(rechargenodes) + 1:length(neighbors) + length(rechargenodes) + length(dirichletnodes)]
		@show maximum(thesehycos), extrema(thesesources), extrema(thesedirichletheads)
		u, of, gradient = r3adjoint(neighbors, areasoverlengths, thesehycos, thesesources, dirichletnodes, thesedirichletheads)
		return of
	end

	function gradient!(xunbounded, storage)
		global neighbors
		global rechargenodes
		global dirichletnodes
		global numnodes
		x = atantransform(xunbounded, lowerbounds, upperbounds)
		thesehycos = exp.(x[1:length(neighbors)])
		thesesources = zeros(numnodes)
		for (i, node) in enumerate(rechargenodes)
			thesesources[node] = x[length(neighbors) + i]
		end
		thesedirichletheads = x[length(neighbors) + length(rechargenodes) + 1:length(neighbors) + length(rechargenodes) + length(dirichletnodes)]
		@show extrema(thesehycos), extrema(thesesources), extrema(thesedirichletheads)
		u, of, gradient = r3adjoint(neighbors, areasoverlengths, thesehycos, thesesources, dirichletnodes, thesedirichletheads)
		storage[1:length(neighbors)] = gradient[1:length(neighbors)] .* thesehycos
		for (i, node) in enumerate(rechargenodes)
			storage[length(neighbors) + i] = gradient[length(neighbors) + node]
		end
		storage[length(neighbors) + length(rechargenodes) + 1:length(neighbors) + length(rechargenodes) + length(dirichletnodes)] = gradient[length(neighbors) + numnodes + 1:end]
		atantransformgradient!(storage, xunbounded, lowerbounds, upperbounds)
		maxgradhyco = maximum(abs.(storage[1:length(neighbors)]))
		maxgradsources = maximum(abs.(storage[length(neighbors) + 1:length(neighbors) + length(rechargenodes)]))
		maxgradbc = maximum(abs.(storage[length(neighbors) + length(rechargenodes) + 1:end]))
		@show maxgradhyco, maxgradsources, maxgradbc
	end
	@time opt = Optim.optimize(objfunc, gradient!, x0unbounded, Optim.LBFGS(), Optim.Options(iterations=175, show_trace=true))
end
xunbounded = opt.minimizer
x = atantransform(xunbounded, lowerbounds, upperbounds)
opthycos = exp.(x[1:length(neighbors)])
optrecharge = x[length(neighbors) + 1:length(neighbors) + length(rechargenodes)]
optsources = zeros(numnodes)
for (i, node) in enumerate(rechargenodes)
	optsources[node] = optrecharge[i]
end
optdirichletheads = x[length(neighbors) + length(rechargenodes) + 1:length(neighbors) + length(rechargenodes) + length(dirichletnodes)]
u, of, gradient = r3adjoint(neighbors, areasoverlengths, opthycos, optsources, dirichletnodes, optdirichletheads)
of = 0.0
opthead, freenode, nodei2freenodei = FiniteVolume.freenodes2nodes(u, optsources, dirichletnodes, optdirichletheads)
@show norm(obsvalues - opthead[obsnodes])^2

x0, x1 = extrema(xs)
y0, y1 = extrema(ys)
z0, z1 = extrema(zs)
numxgridpoints = 2000
numygridpoints = 2000
gridxs = [x for x in linspace(x0, x1, numxgridpoints), y in linspace(y0, y1, numygridpoints)]
gridys = [y for x in linspace(x0, x1, numxgridpoints), y in linspace(y0, y1, numygridpoints)]
if !isdefined(:gridwl)
	@time gridwl = interp.griddata((xs[topnodes], ys[topnodes]), opthead[topnodes], (gridxs, gridys), method="linear")
	#@time initgridwl = interp.griddata((xs[topnodes], ys[topnodes]), adjointhead[topnodes], (gridxs, gridys), method="linear")
	@time gridsources = interp.griddata((xs[topnodes], ys[topnodes]), optsources[topnodes], (gridxs, gridys), method="nearest")
	@time gridhycos = interp.griddata((hycoxs[tophyconodes], hycoys[tophyconodes]), log10.(opthycos[tophyconodes]), (gridxs, gridys), method="nearest")
end

function setfigup()
	fig, ax = PyPlot.subplots(figsize=(16,9), dpi=240)
end
function shutfigdown(fig, ax, filename)
	ax[:plot](obsxs, obsys, "k.", ms=0.5)
	for i = 1:length(obsxs)
		ax[:text](obsxs[i] + 10, obsys[i] + 10, obsnames[i], fontsize=2)
	end
	ax[:set_aspect]("equal", "datalim")
	fig[:tight_layout]()
	if filename != nothing
		fig[:savefig](filename)
	end
	display(fig); println()
	PyPlot.close(fig)
end
function imshow(data, filename=nothing)
	fig, ax = setfigup()
	#img = ax[:imshow](map(x->min(max(x, -5.2), -4.8), data'), origin="lower", extent=[x0, x1, y0, y1])
	img = ax[:imshow](data', origin="lower", extent=[x0, x1, y0, y1])
	fig[:colorbar](img)
	shutfigdown(fig, ax, filename)
end
function contour(data, filename=nothing)
	fig, ax = setfigup()
	cs = ax[:contour](gridxs, gridys, data, 400, linewidths=0.25)
	cs[:clabel](inline=1, fontsize=5)#, fmt="%1.1f")
	shutfigdown(fig, ax, filename)
end
imshow(gridhycos, "hyco.png")
imshow(gridsources, "sources.png")
contour(gridwl, "wls.pdf")

errormatrix = hcat(obsnames, abs(obsvalues - opthead[obsnodes]), obsvalues - opthead[obsnodes])
sortedindices = sort(1:size(errormatrix, 1), by=i->errormatrix[i, 2])
errormatrix = errormatrix[sortedindices, :]
for i = 1:size(errormatrix, 1)
	println(join(errormatrix[i, :], "\t"))
end
