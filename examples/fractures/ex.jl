import FiniteVolume
import JLD

meshdir = "fourfractures"
xs, ys, zs, neighbors, areasoverlengths, fractureindices, dirichletnodes, dirichletheads = JLD.load(joinpath(meshdir, "mesh.jld"), "xs", "ys", "zs", "neighbors", "areasoverlengths", "fractureindices", "dirichletnodes", "dirichletheads")
sources = zeros(length(xs))
conductivities = ones(length(neighbors))
h, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads)
hpflotran = JLD.load(joinpath(meshdir, "pflotran_solution.jld"), "h")
@show norm(A * h[freenode] - b)
@show norm(A * hpflotran[freenode] - b)
nothing
