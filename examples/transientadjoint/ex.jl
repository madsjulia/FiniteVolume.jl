#=
if nprocs() == 1
	#machines = ["mads03", "mads04", "mads05", "mads06"]
	machines = ["mads$(lpad(i, 2, 0))" for i = 6:9]
	nprocs_per_machine = 64
	machinenames = []
	for i = 1:nprocs_per_machine
		machinenames = [machinenames; machines]
	end
	addprocs(machinenames)
end
=#

Distributed.@everywhere begin
	import Distributed
	import FiniteVolume
	import GaussianRandomFields
	import Interpolations
	import LinearAlgebra
	import MatrixFreeHessianOptimization
	import QuadGK
	import PyPlot
	import Random
	import Statistics
	using LaTeXStrings

	Random.seed!(0)
	dt0 = 60.0
	atol = 1e0
	steadyhead = 1e1
	sidelength = 50.0#m
	thickness = 10.0#m
	mins = [-sidelength, -sidelength, 0]
	maxs = [sidelength, sidelength, thickness]
	ns = [20, 20, 5]
	meanloghyco = log(1e-5)#m/s
	Q = 1e-3#m^3/s
	Ss = 0.1#m^-1
	const sigmaval = 1e-1
	sigma = (i, t)->sigmaval#m
	coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)

	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	zs = range(mins[3]; stop=maxs[3], length=ns[3])
	grf = GaussianRandomFields.GaussianRandomField(GaussianRandomFields.CovarianceFunction(3, GaussianRandomFields.Matern(10.0, 2.0)), GaussianRandomFields.CirculantEmbedding(), xs, ys, zs)
	loghycos = GaussianRandomFields.sample(grf) .+ meanloghyco
	dirichletnodes = Int[]
	dirichletheads = Float64[]
	for i = 1:size(coords, 2)
		if abs(coords[1, i]) == sidelength || abs(coords[2, i]) == sidelength
			push!(dirichletnodes, i)
			push!(dirichletheads, steadyhead)
		end
	end

	u0 = fill(steadyhead, size(coords, 2))
	t0 = 0.0
	t1 = 60 * 60 * 24 * 1e1
	tspan = (t0, t1)
	p0 = fill(meanloghyco, length(loghycos))
	p_true = loghycos[1:end]
	sqrtnumobsnodes = 3
	obsxs = range(-sidelength; stop=sidelength, length=sqrtnumobsnodes + 2)[2:end - 1]
	obsys = range(-sidelength; stop=sidelength, length=sqrtnumobsnodes + 2)[2:end - 1]
	obszs = range(0; stop=thickness, length=4)[2:end - 1]
	numobsnodes = length(obsxs) * length(obsys) * length(obszs)
	uobss = fill(t->zeros(length(volumes)), numobsnodes)#we need to define uobs before calling f_and_g_plus
	obsnodes = Array{Int}(undef, numobsnodes)
	obsnodecount = 1
	for z in obszs, y in obsys, x in obsxs
		global obsnodecount
		obsnodes[obsnodecount] = FiniteVolume.getnearestgridpoint([x, y, z], coords)
		obsnodecount += 1
	end
	freenodes, nodei2freenodei = FiniteVolume.getfreenodes(length(u0), dirichletnodes)
	freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
	obsfreenodes = map(x->nodei2freenodei[x], obsnodes)
	hycoregmat = FiniteVolume.hycoharmonicregularizationmatrix(neighbors, areasoverlengths)
	hycoregularization = 1e1

	function f_and_g_plus(p; savecontinuoussolutions=true)
		@assert length(p) == size(coords, 2)
		@time begin
			global uobss
			sources = zeros(size(coords, 2))
			conductivities = FiniteVolume.nodehycos2neighborhycos(neighbors, p, true)
			bigp = [conductivities; sources; dirichletheads]
			l, h = extrema(conductivities)
			if exp(h - l) > 10^6#if the contrast is too large, just get out without even trying to run the model
				println("quick exit $(exp(h - l))")
				return Inf, fill(NaN, length(conductivities)), nothing, nothing
			end
			local us_p
			local ts_p
			uc_ps = []
			uc_p2s = []
			dGdp = zeros(length(p))
			dataof = 0.0
			for i = 1:length(obsnodes)
				sources[obsnodes[i]] = -Q
				bpumping = FiniteVolume.assembleb(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, i->i, true)
				FiniteVolume.scalebyvolume!(bpumping, Ss * volumes, freenodei2nodei)
				bidle = FiniteVolume.assembleb(neighbors, areasoverlengths, conductivities, zeros(length(sources)), dirichletnodes, dirichletheads, i->i, true)
				FiniteVolume.scalebyvolume!(bidle, Ss * volumes, freenodei2nodei)
				getb(t) = t < 60 * 60 * 24 * 3 ? bpumping : bidle
				us_p, ts_p = FiniteVolume.backwardeulerintegrate(u0, tspan, getb, Ss, volumes, neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0)
				uc_p = FiniteVolume.getcontinuoussolution(us_p, ts_p)
				uc_p2 = FiniteVolume.getcontinuoussolution(us_p, ts_p, Val{2})
				if savecontinuoussolutions
					push!(uc_ps, uc_p)
					push!(uc_p2s, uc_p2)
				end
				g, dgdu, dfdp, dgdp, du0dp, G = FiniteVolume.getadjointfunctions(sigma, obsfreenodes, uobss[i], u0, tspan, Ss, volumes, neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0)
				lambdas, ts_lambda = FiniteVolume.adjointintegrate(t->dgdu(uc_p, t), tspan, Ss, volumes, neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0)
				idfdplambda = FiniteVolume.integratedfdplambda(uc_p2, bigp, lambdas, ts_lambda, tspan, Ss, volumes, neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads, i->i, true)
				thisgradient = FiniteVolume.gradientintegrate(FiniteVolume.getcontinuoussolution(lambdas, ts_lambda), du0dp, t->dgdp(uc_p, t, bigp), t->dfdp(uc_p, t, bigp), tspan; maxevals=3 * 10^2, order=21)
				for i = 1:length(neighbors)#do the chain rule to convert the gradient of the neighbor hycos to the gradient of the node hycos
					dGdp[neighbors[i][1]] += 0.5 * thisgradient[i]
					dGdp[neighbors[i][2]] += 0.5 * thisgradient[i]
				end
				dataof += G(uc_p)
			end
			hycoregof = hycoregularization * LinearAlgebra.norm(hycoregmat * p)^2
			dGdp += (2 * hycoregularization) * transpose(hycoregmat) * (hycoregmat * p)
		end
		return dataof + hycoregof, dGdp, uc_p2s, uc_ps
	end

	function f_and_g(p)
		a, b, c, d = f_and_g_plus(p; savecontinuoussolutions=false)
		return a, b
	end
	G_true, dGdp_true, uobs2s, uobss = f_and_g_plus(p_true)
end

@time G_p0, dGdp_p0, uc_p02s, _ = f_and_g_plus(p0)

#check that the gradient is legit
deltap = 1e-6
p0pd = copy(p0)
p0pd[argmax(abs.(dGdp_p0))] += deltap
G_p0pd, _ = f_and_g(p0pd)
@show dGdp_p0[argmax(abs.(dGdp_p0))]
@show (G_p0pd - G_p0) / deltap

function hessian_callback(H, iter)
	fig, ax = PyPlot.subplots()
	eigvals = sort(abs.(H.v); rev=true)
	ax[:plot](eigvals, "k.")
	ax[:set_xlabel](L"i")
	ax[:set_ylabel](L"|\lambda_i|")
	display(fig)
	fig[:savefig]("eigenfigs/eigvals_$iter.pdf")
	println()
	PyPlot.close(fig)
end

#@time p_opt, G_opt = MatrixFreeHessianOptimization.quasinewton(f_and_g, p0, div(nworkers(), 2); lambda_mu=sqrt(2.0), maxIter=1, np_lambda=32, show_trace=true, hessian_callback=hessian_callback, lambda=100.0)
#@time p_opt, G_opt = MatrixFreeHessianOptimization.quasinewton(f_and_g, p0, div(nworkers(), 2) - 4; lambda_mu=sqrt(10.0), maxIter=5, np_lambda=16, show_trace=true, hessian_callback=hessian_callback, lambda=1e4)
@time p_opt, G_opt = MatrixFreeHessianOptimization.quasinewton(f_and_g, p0, 60; lambda_mu=sqrt(10.0), maxIter=10, np_lambda=12, show_trace=true, hessian_callback=hessian_callback, lambda=1e4)
@time G_opt, dGdp_opt, uc_opt2s, _ = f_and_g_plus(p_opt)

function plotfits(wellnum=div(length(uobs2s), 2); saveresults=false)
	uc_p02 = uc_p02s[wellnum]
	uobs2 = uobs2s[wellnum]
	uc_opt2 = uc_opt2s[wellnum]
	fig, axs = PyPlot.subplots(sqrtnumobsnodes, 2 * sqrtnumobsnodes, sharex=true, sharey=false, figsize=(16, 9))
	ts = range(tspan[1]; stop=tspan[2], length=101)
	for i = 1:numobsnodes
		ax = axs[i]
		ax[:plot](ts, map(t->uc_p02(obsnodes[i], t), ts), label="u(p0)", alpha=0.5)
		ax[:plot](ts, map(t->uc_opt2(obsnodes[i], t), ts), label="opt", alpha=0.5)
		ax[:plot](ts, map(t->uobs2(obsnodes[i], t), ts), label="obs", alpha=0.5)
	end
	axs[end][:legend]()
	fig[:tight_layout]()
	if saveresults
		fig[:savefig]("heads.png")
	end
	display(fig)
	println()
	PyPlot.close(fig)
end

plotfits(; saveresults=true)

function plothycos(zfrac=0.5)
	hycoitp = Interpolations.interpolate((xs, ys, zs), reshape(p_true, length(xs), length(ys), length(zs)), Interpolations.Gridded(Interpolations.Linear()))
	hycoitp_opt = Interpolations.interpolate((xs, ys, zs), reshape(p_opt, length(xs), length(ys), length(zs)), Interpolations.Gridded(Interpolations.Linear()))
	plotxs = range(mins[1]; stop=maxs[1], length=50)
	plotys = range(mins[2]; stop=maxs[2], length=50)
	hyco_imgdata = Array{Float64}(undef, length(plotys), length(plotxs))
	hyco_imgdata_opt = Array{Float64}(undef, length(plotys), length(plotxs))
	for (i, x) in enumerate(plotxs)
		for (j, y) in enumerate(plotys)
			hyco_imgdata[size(hyco_imgdata, 1) + 1 - j, i] = hycoitp(x, y, zfrac * thickness)
			hyco_imgdata_opt[size(hyco_imgdata, 1) + 1 - j, i] = hycoitp_opt(x, y, zfrac * thickness)
		end
	end
	fig, axs = PyPlot.subplots(1, 2, figsize=(16,9))
	ax = axs[1]
	ax[:imshow](hyco_imgdata, extent=[minimum(xs), maximum(xs), minimum(ys), maximum(ys)])
	ax[:plot](coords[1, obsnodes], coords[2, obsnodes], "r.", alpha=0.5)
	ax = axs[2]
	ax[:imshow](hyco_imgdata_opt, extent=[minimum(xs), maximum(xs), minimum(ys), maximum(ys)])
	ax[:plot](coords[1, obsnodes], coords[2, obsnodes], "r.", alpha=0.5)
	fig[:savefig]("perms.png")
	display(fig)
	println()
	PyPlot.close(fig)
end

plothycos()

function plothycosraw(zind)
	hyco_imgdata = reshape(p_true, length(xs), length(ys), length(zs))[:, :, zind]
	hyco_imgdata_opt = reshape(p_opt, length(xs), length(ys), length(zs))[:, :, zind]
	fig, axs = PyPlot.subplots(1, 2, figsize=(16,9))
	ax = axs[1]
	ax[:imshow](hyco_imgdata, extent=[minimum(xs), maximum(xs), minimum(ys), maximum(ys)])
	ax[:plot](coords[1, obsnodes], coords[2, obsnodes], "r.", alpha=0.5)
	ax = axs[2]
	ax[:imshow](hyco_imgdata_opt, extent=[minimum(xs), maximum(xs), minimum(ys), maximum(ys)])
	ax[:plot](coords[1, obsnodes], coords[2, obsnodes], "r.", alpha=0.5)
	fig[:savefig]("perms.png")
	display(fig)
	println()
	PyPlot.close(fig)
end

#=
plothycosraw(1)
plothycosraw(ns[3])
=#
for i = 1:ns[3]
	plothycosraw(i)
end

@show Statistics.cor(p_opt, p_true)
