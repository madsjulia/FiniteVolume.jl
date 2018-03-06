using Base.Test
import FiniteVolume

xs = [0.0 1/3 2/3 1.0;
	  0.0 0.0 0.0 0.0;
	  0.0 0.0 0.0 0.0]
neighbors = [1=>2, 2=>1, 2=>3, 3=>2, 3=>4, 4=>3]
areasoverlengths = ones(length(neighbors))
conductivities = deepcopy(areasoverlengths)
sources = zeros(size(xs, 2))
volumes = ones(size(xs, 2))
dirichletnodes = [1, 4]
dirichletvalues = [1.0,  0.0]

h, ch, A, B, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletvalues)
@test h ≈ [1.0, 2/3, 1/3, 0.0]

include("ode.jl")
include("odeadjoint.jl")
include("theis.jl")
include("onenodeadjoint.jl")
