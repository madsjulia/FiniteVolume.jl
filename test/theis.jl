using Test
import FiniteVolume
import LinearAlgebra

#this approximation of W(u) is derived from http://www.kgs.ku.edu/Publications/Bulletins/GW3/
function W(u)
	if u <= 1
		return -log(u) + -0.57721566 + 0.99999193u^1 + -0.24991055u^2 + 0.05519968u^3 + -0.00976004u^4 + 0.00107857u^5
	else
		return (u^2 + 2.334733u + 0.250621) / (u^2 + 3.330657u + 1.681534) * exp(-u) / u
	end
end
function theisdrawdown(t::Number, r::Number, T::Number, S::Number, Q::Number)
	return Q * W(r^2 * S / (4 * T * t)) / (4 * pi * T)
end

function thiemdrawdown(r, T, Q, R)
	return Q * log(R / r) / (2 * pi * T)
end

steadyhead = 1e3
sidelength = 50.0
thickness = 10.0
mins = [-sidelength, -sidelength, 0]
maxs = [sidelength, sidelength, thickness]
ns = [101, 101, 2]
k = 1e-5
Q = 1e-3
Ss = 0.1
S = Ss * thickness
coords, neighbors, areasoverlengths, volumes = FiniteVolume.regulargrid(mins, maxs, ns)
hycos = fill(k, length(areasoverlengths))
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

usteady, _, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
u0 = fill(steadyhead, size(coords, 2))
us, ts = FiniteVolume.backwardeulerintegrate(u0, (0.0, 60 * 60 * 24 * 1e1), Ss, volumes, neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads; atol=1e-4, dt0=60.0)

r0 = 0.1
goodnodes = collect(filter(i->coords[3, i] == thickness && coords[2, i] == 0 && coords[1, i] > r0 && coords[1, i] <= sidelength / 2, 1:size(coords, 2)))
rs = coords[1, goodnodes]
T = thickness * k
theisdrawdowns = theisdrawdown.(ts[end], rs, T, S, Q)
modeldrawdowns = -us[end][goodnodes] + steadyhead
@test isapprox(theisdrawdowns, modeldrawdowns, atol=1e-4, rtol=2e-2)
thiemdrawdowns = thiemdrawdown.(rs, T, Q, sidelength)
steadydrawdowns = -usteady[goodnodes] + steadyhead
@test isapprox(thiemdrawdowns, steadydrawdowns, atol=1e-4, rtol=2e-2)
