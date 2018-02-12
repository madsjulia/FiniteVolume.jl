import FiniteVolume
import JLD
import PyAMG
import RecycledCG

verbosity = false

meshdir = "backbone_x01"
@time xs, ys, zs, neighbors, areasoverlengths, fractureindices, dirichletnodes, dirichletheads = JLD.load(joinpath(meshdir, "mesh.jld"), "xs", "ys", "zs", "neighbors", "areasoverlengths", "fractureindices", "dirichletnodes", "dirichletheads")
sources = zeros(length(xs))
conductivities = ones(length(neighbors))
node2fracture = Dict(zip(1:length(xs), fractureindices))
connection2fracture = Dict(zip(1:length(conductivities), map(p->node2fracture[p[1]], neighbors)))
metaindex(i) = connection2fracture[i]
@time h, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, conductivities, sources, dirichletnodes, dirichletheads)
@time begin
	M = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(A))
	Ar = RecycledCG.MatrixWithRecycling(A)
	hrcg, isconverged, numiters = RecycledCG.rcg(Ar, b; M=M, maxiters=50, rtol=1e-8, verbose=verbosity)
	@show isconverged
end
@time begin
	D = FiniteVolume.assembleA(neighbors, areasoverlengths, 0 * conductivities, sources, dirichletnodes, dirichletheads)
	hrecycled = RecycledCG.recycle(Ar, D, b)
end

trecycle = 0.0
tprecond = 0.0
trcg = 0.0
tfull = 0.0
rcgiters = 0
fulliters = 0
srand(0)
changefs = rand(unique(fractureindices), 3)
#changefs = unique(fractureindices)
for i = 1:length(changefs)
	newconductivities = map(p->ifelse(node2fracture[p[1]] == changefs[i] && node2fracture[p[2]] == changefs[i], -0.5, 0.0), neighbors)
	D = FiniteVolume.assembleA(neighbors, areasoverlengths, newconductivities, sources, dirichletnodes, dirichletheads)
	newb = b + FiniteVolume.assembleb(neighbors, areasoverlengths, newconductivities, sources, dirichletnodes, dirichletheads)
	trecycle += @elapsed hrecycled2 = RecycledCG.recycle(Ar, D, newb, hrcg)
	#trecycle += @elapsed hrecycled2 = RecycledCG.recycle(Ar, D, newb)
	newA = A + D
	@show norm(newA * hrecycled2 - newb)
	#tprecond += @elapsed newM = PyAMG.aspreconditioner(PyAMG.RugeStubenSolver(newA))
	newM = M
	trcg += @elapsed hrecycled2pp, isconverged, numiters = RecycledCG.rcg(newA, newb, hrecycled2; M=newM, maxiters=50, rtol=0.0, atol=1e-5, verbose=verbosity)
	rcgiters += numiters
	tfull += @elapsed _, isconverged, numiters = RecycledCG.rcg(newA, newb; M=newM, maxiters=50, rtol=0.0, atol=1e-5, verbose=verbosity)
	fulliters += numiters
	#=
	@show numiters
	@time FiniteVolume.solvediffusion(neighbors, areasoverlengths, newconductivities + conductivities, sources, dirichletnodes, dirichletheads)
	@show norm(newA * hrecycled2 - newb)
	@show norm(newA * hrecycled2pp - newb)
	@show norm(newb)
	=#
end


@show norm(A * h[freenode] - b)
@show norm(A * hrcg - b)
@show norm(A * hrecycled - b)
@show trecycle + tprecond + trcg
@show tprecond + tfull
@show rcgiters
@show fulliters
