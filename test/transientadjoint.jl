using Test
import FiniteVolume
import GaussianRandomFields
import Interpolations
import LinearAlgebra

srand(0)
doplot = false

atol = 1e-2
steadyhead = 1e3
sidelength = 50.0#m
thickness = 10.0#m
mins = [-sidelength, -sidelength, 0]
maxs = [sidelength, sidelength, thickness]
ns = [51, 51, 2]
meanloghyco = log(1e-5)#m/s
Q = 1e-3#m^3/s
Ss = 0.1#m^-1
const sigma = 0.03#m
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
	if LinearAlgebra.norm(coords[1:2, i]) - sidelength >= 0
		push!(dirichletnodes, i)
		push!(dirichletheads, steadyhead)
	end
end

u0 = fill(steadyhead, size(coords, 2))
t0 = 0.0
t1 = 60 * 60 * 24 * 1e1
tspan = (t0, t1)
@time us, ts = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, loghycos, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=60.0)
uobs = FiniteVolume.getcontinuoussolution(map(x->x + sigma * randn(size(x)), us), ts)
uc = FiniteVolume.getcontinuoussolution(us, ts)
p0 = [fill(meanloghyco, length(loghycos)); sources; dirichletheads]
@time us_init, ts_init = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, fill(meanloghyco, length(loghycos)), sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=60.0)
uc_init = FiniteVolume.getcontinuoussolution(us_init, ts_init)

numobsnodes = 100
freenodes, nodei2freenodei = FiniteVolume.getfreenodes(length(u0), dirichletnodes)
freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
obsfreenodes = randperm(sum(freenodes))[1:numobsnodes]
obsnodes = map(x->freenodei2nodei[x], obsfreenodes)

function g(u, t, p, uobs)
	global sigma
	uobseval = uobs(t)
	ueval = u(t)
	retval = 0.0
	for i in obsfreenodes
		retval += sigma^2 * (ueval[freenodei2nodei[i]] - uobseval[freenodei2nodei[i]])^2
	end
	return retval
end

function dgdu(u, t, p, uobs)
	global sigma
	uobseval = uobs(t)
	ueval = u(t)
	result = zeros(sum(freenodes))
	for i in obsfreenodes
		result[i] = 2 * sigma^2 * (ueval[freenodei2nodei[i]] - uobseval[freenodei2nodei[i]])
	end
	return result
end

dgdp(u, t, p, uobs) = zeros(length(p))

function dfdp(u, t, p, uobs)
	ueval = u(t)[freenodes]
	p_loghycos = p[1:length(loghycos)]
	p_sources = p[length(loghycos) + 1:length(loghycos) + length(sources)]
	p_dirichletheads = p[length(loghycos) + length(sources) + 1:length(loghycos) + length(sources) + length(dirichletheads)]
	A_px = FiniteVolume.assembleA_px(ueval, neighbors, areasoverlengths, p_loghycos, p_sources, dirichletnodes, p_dirichletheads, i->i, true)
	b_p = FiniteVolume.assembleb_p(neighbors, areasoverlengths, p_loghycos, p_sources, dirichletnodes, p_dirichletheads, i->i, true)
	result = transpose(b_p - A_px)
	FiniteVolume.scalebyvolume!(result, Ss * volumes, freenodei2nodei)
	return transpose(result)
end

@test g(uobs, 0.5 * t1, p0, uobs) == 0
@test dgdu(t->uobs(t) + 1, 0.5 * t1, p0, uobs) == [i in obsfreenodes ? 2 * sigma^2 : 0.0 for i = 1:sum(freenodes)]

function G(p)
	p_loghycos = p[1:length(loghycos)]
	p_sources = p[length(loghycos) + 1:length(loghycos) + length(sources)]
	p_dirichletheads = p[length(loghycos) + length(sources) + 1:length(loghycos) + length(sources) + length(dirichletheads)]
	us_p, ts_p = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, p_loghycos, p_sources, dirichletnodes, p_dirichletheads, i->i, true; atol=atol, dt0=60.0)
	uc_p = FiniteVolume.getcontinuoussolution(us_p, ts_p)
	I, E = QuadGK.quadgk(t->g(uc_p, t, p, uobs), tspan...; maxevals=3*10^2, order=21)
	return I
end

@time lambdas, ts_lambda = FiniteVolume.adjointintegrate(t->dgdu(uc_init, t, p0, uobs), tspan, Ss, volumes, neighbors, areasoverlengths, fill(meanloghyco, length(loghycos)), sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=60.0)
lambdac = FiniteVolume.getcontinuoussolution(lambdas, ts_lambda)
du0dp = spzeros(length(p0), sum(freenodes))
@time dGdp, E = FiniteVolume.gradientintegrate(lambdac, du0dp, t->dgdp(uc_init, t, p0, uobs), t->dfdp(uc_init, t, p0, uobs), tspan; maxevals=3*10^2, order=21)
#importantindices = sort(1:length(loghycos); by=i->abs(dGdp[i]), rev=true)[1:20]
importantindices = sort(1:length(dGdp); by=i->abs(dGdp[i]), rev=true)[1:20]
deltap = 1e-6
factors = Float64[]
for i in importantindices
	p0pd = copy(p0)
	p0pd[i] += deltap
	x1 = (G(p0pd) - G(p0)) / deltap
	x2 = dGdp[i]
	@show x1, x2, x2 / x1
	push!(factors, x2 / x1)
end
@show deltap, median(factors), std(factors)

if doplot
	itp = Interpolations.interpolate((ts,), us, Interpolations.Gridded(Interpolations.Linear()))
	hycoitp = Interpolations.interpolate((xs, ys, zs), reshape(log10.(nodeloghycos), length(xs), length(ys), length(zs)), Interpolations.Gridded(Interpolations.Linear()))
	kgradient = FiniteVolume.neighborhycos2nodehycos(neighbors, dGdp[1:length(loghycos)], ns)
	graditp = Interpolations.interpolate((xs, ys, zs), reshape(kgradient, length(xs), length(ys), length(zs)), Interpolations.Gridded(Interpolations.Linear()))
	xs = linspace(mins[1], maxs[1], 500)
	ys = linspace(mins[2], maxs[2], 500)
	himgdata = Array{Float64}(length(ys), length(xs))
	kimgdata = Array{Float64}(length(ys), length(xs))
	gradimgdata = Array{Float64}(length(ys), length(xs))
	for (framenum, t) in enumerate(linspace(t0, t1, 3))
		u = FiniteVolume.makeinterpolant(mins, maxs, ns, uc(t))
		for (i, x) in enumerate(xs)
			for (j, y) in enumerate(ys)
				v = steadyhead - u(x, y, 0.5 * thickness)
				himgdata[size(himgdata, 1) + 1 - j, i] = v > 1e-3 ? log10(v) : NaN
				kimgdata[size(himgdata, 1) + 1 - j, i] = v > 1e-9 ? hycoitp[x, y, 0.5 * thickness] : NaN
				gradimgdata[size(himgdata, 1) + 1 - j, i] = v > 1e-9 ? graditp[x, y, 0.5 * thickness] : NaN
			end
		end
		fig, axs = PyPlot.subplots(1, 3, figsize=(16,9))
		ax = axs[1]
		cax = ax[:imshow](himgdata, extent=[minimum(xs), maximum(xs), minimum(ys), maximum(ys)], vmin=-3, vmax=log10.(maximum(steadyhead - us[end])))
		ax[:plot](coords[1, obsnodes], coords[2, obsnodes], "r.", alpha=0.5)
		ax = axs[2]
		cax = ax[:imshow](kimgdata, extent=[minimum(xs), maximum(xs), minimum(ys), maximum(ys)])
		ax[:plot](coords[1, obsnodes], coords[2, obsnodes], "r.", alpha=0.5)
		ax = axs[3]
		cax = ax[:imshow](gradimgdata, extent=[minimum(xs), maximum(xs), minimum(ys), maximum(ys)])
		ax[:plot](coords[1, obsnodes], coords[2, obsnodes], "r.", alpha=0.5)
		display(fig)
		println()
		#fig[:savefig]("figs/frame_$(lpad(framenum, 4, 0)).png")
		PyPlot.close(fig)
	end
	#run(`ffmpeg -i figs/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p -f mp4 -y figs/drawdown.mp4`)
end
