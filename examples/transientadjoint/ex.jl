using Base.Test
import FiniteVolume
import GaussianRandomFields
import Interpolations
import MatrixFreeHessianOptimization
import QuadGK
import PyPlot

@everywhere begin
	srand(0)
	dt0 = 60.0
	atol = 1e-2
	steadyhead = 1e1
	sidelength = 50.0#m
	thickness = 10.0#m
	mins = [-sidelength, -sidelength, 0]
	maxs = [sidelength, sidelength, thickness]
	ns = [51, 51, 2]
	meanloghyco = log(1e-5)#m/s
	Q = 1e-3#m^3/s
	Ss = 0.1#m^-1
	const sigmaval = 1e-1
	sigma = (i, t)->sigmaval#m
	coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)

	xs = linspace(mins[1], maxs[1], ns[1])
	ys = linspace(mins[2], maxs[2], ns[2])
	zs = linspace(mins[3], maxs[3], ns[3])
	grf = GaussianRandomFields.GaussianRandomField(GaussianRandomFields.CovarianceFunction(3, GaussianRandomFields.Matern(10.0, 2.0)), GaussianRandomFields.CirculantEmbedding(), xs, ys, zs)
	nodeloghycos = GaussianRandomFields.sample(grf) + meanloghyco
	loghycos = FiniteVolume.nodehycos2neighborhycos(neighbors, nodeloghycos, true)
	sources = zeros(size(coords, 2))
	centerindices = Int[]
	for i = 1:size(coords, 2)
		if coords[1, i] == 0 && coords[2, i] == 0
			push!(centerindices, i)
		end
	end
	sources[centerindices[1]] = -Q / (2 * length(centerindices) - 2)
	sources[centerindices[end]] = -Q / (2 * length(centerindices) - 2)
	sources[centerindices[2:end - 1]] = -2 * Q / (2 * length(centerindices) - 2)
	dirichletnodes = Int[]
	dirichletheads = Float64[]
	for i = 1:size(coords, 2)
		if norm(coords[1:2, i]) - sidelength >= 0
			push!(dirichletnodes, i)
			push!(dirichletheads, steadyhead)
		end
	end

	u0 = fill(steadyhead, size(coords, 2))
	t0 = 0.0
	t1 = 60 * 60 * 24 * 1e1
	tspan = (t0, t1)
	@time us, ts = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, loghycos, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0)
	uobs = FiniteVolume.getcontinuoussolution(us, ts)
	uobs2 = FiniteVolume.getcontinuoussolution(us, ts, Val{2})
	p0 = fill(meanloghyco, length(loghycos))
	p_true = loghycos
	sqrtnumobsnodes = 5
	numobsnodes = sqrtnumobsnodes^2
	freenodes, nodei2freenodei = FiniteVolume.getfreenodes(length(u0), dirichletnodes)
	freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
	obsfreenodes = randperm(sum(freenodes))[1:numobsnodes]
	obsnodes = map(x->freenodei2nodei[x], obsfreenodes)
	hycocoords = FiniteVolume.gethycocoords(neighbors, coords)
	hycoregmat = FiniteVolume.knnregularization(hycocoords, 25)
	hycoregularization = 1e-1

	function f_and_g_plus(p)
		bigp = [p; sources; dirichletheads]
		p_conductivities = p[1:length(loghycos)]
		starttime = now()
		function callback()
			if now() - starttime > Dates.Second(10)
				error("timeout")
			end
		end
		local us_p
		local ts_p
		try
			us_p, ts_p = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, p_conductivities, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0, callback=callback)
		catch
			println("timeout 1")
			return Inf, fill(NaN, length(p_conductivities)), nothing
		end
		uc_p = FiniteVolume.getcontinuoussolution(us_p, ts_p)
		uc_p2 = FiniteVolume.getcontinuoussolution(us_p, ts_p, Val{2})
		g, dgdu, dfdp, dgdp, du0dp, G = FiniteVolume.getadjointfunctions(sigma, obsfreenodes, uobs, u0, tspan, Ss, volumes, neighbors, areasoverlengths, p_conductivities, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0)
		local lambdas
		local ts_lambda
		try
			lambdas, ts_lambda = FiniteVolume.adjointintegrate(t->dgdu(uc_p, t), tspan, Ss, volumes, neighbors, areasoverlengths, p_conductivities, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0, callback=callback)
		catch
			println("timeout 2")
			return Inf, fill(NaN, length(p_conductivities)), nothing
		end
		idfdplambda = FiniteVolume.integratedfdplambda(uc_p2, bigp, lambdas, ts_lambda, tspan, Ss, volumes, neighbors, areasoverlengths, p_conductivities, sources, dirichletnodes, dirichletheads, i->i, true)
		dGdp = FiniteVolume.gradientintegrate(FiniteVolume.getcontinuoussolution(lambdas, ts_lambda), du0dp, t->dgdp(uc_p, t, bigp), t->dfdp(uc_p, t, bigp), tspan; maxevals=3 * 10^2, order=21)
		hycoregof = hycoregularization * norm(hycoregmat * p_conductivities)^2
		dGdp[1:length(p_conductivities)] += (2 * hycoregularization) * At_mul_B(hycoregmat, (hycoregmat * p_conductivities))
		return G(uc_p) + hycoregof, dGdp[1:length(p_conductivities)], uc_p2
	end

	function f_and_g(p)
		a, b, c = f_and_g_plus(p)
		return a, b
	end
end

@time G_true, dGdp_true, uc_true2 = f_and_g_plus(p_true)
@time G_p0, dGdp_p0, uc_p02 = f_and_g_plus(p0)
@show G_p0

#check that the gradient is legit
i = indmax(abs.(dGdp_p0))
deltap = 1e-6
p0pd = copy(p0)
p0pd[i] += deltap
G_p0pd, _ = f_and_g(p0pd)
@show dGdp_p0[i]
@show (G_p0pd - G_p0) / deltap

function hessian_callback(H, iter)
	fig, ax = PyPlot.subplots()
	eigvals = sort(abs.(H.v); rev=true)
	ax[:plot](eigvals, "k.")
	display(fig)
	println()
	PyPlot.close(fig)
end

@time p_opt, G_opt = MatrixFreeHessianOptimization.quasinewton(f_and_g, p0, 100; maxIter=10, np_lambda=nworkers(), show_trace=true, hessian_callback=hessian_callback)
@time G_opt, dGdp_opt, uc_opt2 = f_and_g_plus(p_opt)

fig, axs = PyPlot.subplots(sqrtnumobsnodes, sqrtnumobsnodes, sharex=true, sharey=true)
ts = linspace(tspan[1], tspan[2], 101)
for i = 1:numobsnodes
	ax = axs[i]
	ax[:plot](ts, map(t->uc_p02[obsnodes[i], t], ts), label="u(p0)", alpha=0.5)
	ax[:plot](ts, map(t->uc_opt2[obsnodes[i], t], ts), label="opt", alpha=0.5)
	ax[:plot](ts, map(t->uobs2[obsnodes[i], t], ts), label="obs", alpha=0.5)
end
axs[end][:legend]()
fig[:tight_layout]
display(fig)
println()
PyPlot.close(fig)
