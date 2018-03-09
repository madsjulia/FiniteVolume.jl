using Base.Test
import FiniteVolume
import GaussianRandomFields
import Interpolations

srand(0)
doplot = false

atol = 1e-4
steadyhead = 0.0
sidelength = 50.0#m
thickness = 10.0#m
mins = [-sidelength, -sidelength, 0]
maxs = [sidelength, sidelength, thickness]
ns = [25, 25, 2]
meanloghyco = log(1e-5)#m/s
Q = 1e-3#m^3/s
Ss = 0.1#m^-1
sigma = (i, t)->0.03#m
coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)

xs = linspace(mins[1], maxs[1], ns[1])
ys = linspace(mins[2], maxs[2], ns[2])
zs = linspace(mins[3], maxs[3], ns[3])
grf = GaussianRandomFields.GaussianRandomField(GaussianRandomFields.CovarianceFunction(3, GaussianRandomFields.Matern(10.0, 2.0)), GaussianRandomFields.CirculantEmbedding(), xs, ys, zs)
nodeloghycos = GaussianRandomFields.sample(grf) + meanloghyco
#loghycos = FiniteVolume.nodehycos2neighborhycos(neighbors, nodeloghycos, true)
loghycos = fill(meanloghyco + 1, length(neighbors))
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
us, ts = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, loghycos, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=60.0)
uobs = FiniteVolume.getcontinuoussolution(us, ts)

p0 = [fill(meanloghyco, length(loghycos)); sources; dirichletheads]
us_init, ts_init = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, fill(meanloghyco, length(loghycos)), sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=60.0)
uc_init = FiniteVolume.getcontinuoussolution(us_init, ts_init)
uc_init2 = FiniteVolume.getcontinuoussolution(us_init, ts_init, Val{2})

freenodes, nodei2freenodei = FiniteVolume.getfreenodes(length(u0), dirichletnodes)
freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
obsnodes = centerindices
obsfreenodes = map(i->nodei2freenodei[i], obsnodes)

g, dgdu, dfdp, dgdp, du0dp, G = FiniteVolume.getadjointfunctions(sigma, obsfreenodes, uobs, u0, tspan, Ss, volumes, neighbors, areasoverlengths, p0[1:length(loghycos)], sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=60.0)
lambdas, ts_lambda = FiniteVolume.adjointintegrate(t->dgdu(uc_init, t), tspan, Ss, volumes, neighbors, areasoverlengths, p0[1:length(loghycos)], sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=60.0)
du0dp = spzeros(length(p0), sum(freenodes))
idfdplambda = FiniteVolume.integratedfdplambda(uc_init2, p0, lambdas, ts_lambda, tspan, Ss, volumes, neighbors, areasoverlengths, p0[1:length(loghycos)], sources, dirichletnodes, dirichletheads, i->i, true)
dGdp = FiniteVolume.gradientintegrate(lambdas[1], du0dp, t->dgdp(uc_init, t, p0), idfdplambda, tspan; maxevals=3 * 10^2, order=21)

importantindices = sort(1:length(dGdp); by=i->abs(dGdp[i]), rev=true)[1:20]
deltap = 1e-4
for i in importantindices
	p0pd = copy(p0)
	p0pd[i] += deltap
	p0md = copy(p0)
	p0md[i] -= deltap
	x1 = (G(p0pd) - G(p0md)) / (2 * deltap)
	x2 = dGdp[i]
	@test isapprox(x1, x2, rtol=1e-3)
end
