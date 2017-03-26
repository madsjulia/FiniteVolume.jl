using Base.Test
import FiniteVolume

xs = [0.0 1/3 2/3 1.0;
	  0.0 0.0 0.0 0.0;
	  0.0 0.0 0.0 0.0]
neighbors = Array{Int, 1}[[2], [1, 3], [2, 4], [3]]
areas = [[1.0], [1.0, 1.0], [1.0, 1.0], [1.0]]
conductivities = deepcopy(areas)
sources = zeros(size(xs, 2))
dirichletnodes = [1, 4]
dirichletvalues = [1.0,  0.0]

h = FiniteVolume.solvediffusion(xs, neighbors, areas, conductivities, sources, dirichletnodes, dirichletvalues)
@test h ≈ [1.0, 2/3, 1/3, 0.0]
