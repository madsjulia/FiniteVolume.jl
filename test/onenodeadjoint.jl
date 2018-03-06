using Base.Test
import FiniteVolume
import GaussianRandomFields
import Interpolations
import PyPlot
import QuadGK

srand(0)
doplot = false

importantindices = [1, 3, 4]
deltap = 1e-8
sigma = (i, t)->0.01
atol = 1e-8
dt0=1e-3
Ss = 1.0
volumes = [1.0, 1.0]
neighbors = [1=>2]
areasoverlengths = [1.0]
loghycos = [0.0]
sources = [0.0, 1.0]
dirichletnodes = Int[1]
dirichletheads = Float64[0.0]
u0 = [0.0, 0.0]
t0 = 0.0
t1 = 1.0
tspan = (t0, t1)
us, ts = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, loghycos, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0)
uobs = FiniteVolume.getcontinuoussolution(us, ts)
uobs_analytical(t) = 1 - exp(-t)
for (i, t) in enumerate(ts)
	@test isapprox(us[i][2], uobs_analytical(t), rtol=1e-4)
end
p0 = [loghycos + 1; sources; dirichletheads]
p0_hycos = loghycos + 1
p0_sources = sources
p0_dirichletheads = dirichletheads
us_init, ts_init = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, p0_hycos, p0_sources, dirichletnodes, p0_dirichletheads, i->i, true; atol=atol, dt0=dt0)
uc_init = FiniteVolume.getcontinuoussolution(us_init, ts_init)
u_init_analytical(t) = (1 - exp(-e * t)) / e
for (i, t) in enumerate(ts_init)
	@test isapprox(us_init[i][2], u_init_analytical(t), rtol=1e-4)
end

freenodes, nodei2freenodei = FiniteVolume.getfreenodes(length(u0), dirichletnodes)
freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
obsnodes = [2]
obsfreenodes = map(i->nodei2freenodei[i], obsnodes)

g, dgdu, dfdp, dgdp, du0dp, G = FiniteVolume.getadjointfunctions(sigma, obsfreenodes, uobs, u0, tspan, Ss, volumes, neighbors, areasoverlengths, loghycos, sources, dirichletnodes, dirichletheads, i->i, true; atol=atol, dt0=dt0)

@test g(uobs, 0.5 * t1, uobs) == 0
@test dgdu(t->uobs(t) + 1, 0.5 * t1, uobs) == [i in obsfreenodes ? 2 * sigma(i, 0.5 * t1)^2 : 0.0 for i = 1:sum(freenodes)]

f_analytical = s->2 * sigma(1, s)^2 * (u_init_analytical(s) - uobs_analytical(s))
lambdas, ts_lambda = FiniteVolume.adjointintegrate(t->dgdu(uc_init, t, uobs), tspan, Ss, volumes, neighbors, areasoverlengths, p0_hycos, p0_sources, dirichletnodes, p0_dirichletheads, i->i, true; atol=atol, dt0=dt0)
lambdac = FiniteVolume.getcontinuoussolution(lambdas, ts_lambda)
gamma_analytical = t->exp(-e * t) * QuadGK.quadgk(s->exp(e * s) * f_analytical(tspan[2] - s), 0, t)[1]
lambda_analytical = t->gamma_analytical(tspan[2] - t)

for (i, t) in enumerate(ts_lambda)
	@test isapprox(lambdas[i][1], lambda_analytical(t), rtol=1e-4, atol=1e-7)
end

dGdp, E = FiniteVolume.gradientintegrate(lambdac, du0dp, t->dgdp(uc_init, t, p0), t->dfdp(uc_init, t, p0), tspan; maxevals=3 * 10^2, order=21)
for i in importantindices
	p0pd = copy(p0)
	p0pd[i] += deltap
	p0md = copy(p0)
	p0md[i] -= deltap
	x1 = (G(p0pd) - G(p0md)) / (2 * deltap)
	x2 = dGdp[i]
	@test isapprox(x1, x2, rtol=1e-2)
end
