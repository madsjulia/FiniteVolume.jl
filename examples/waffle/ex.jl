import FEHM
import FiniteVolume
import PyPlot

westzonenum, westnodes = FEHM.parsezone(joinpath(dirname(@__FILE__), "data/out_west.zonn"))
eastzonenum, eastnodes = FEHM.parsezone(joinpath(dirname(@__FILE__), "data/out_east.zonn"))
wellzonenums, wellzonenodes = FEHM.parsezone(joinpath(dirname(@__FILE__), "data/well_screens.zonn"))
isanode, zoneornodenums, skds, eflows, aipeds = FEHM.parseflow(joinpath(dirname(@__FILE__), "data/wl.flow"))
xs, ys, zs, _ = FEHM.parsegeo(joinpath(dirname(@__FILE__), "data/tet.geo"), false)
r36loc = [5.010628700000E+05, 5.388061300000E+05, 1.765E+03, 1.767E+03, 50]
r36infilnodes = Int[]
for i = 1:length(xs)
	if zs[i] >= r36loc[3] && zs[i] <= r36loc[4]
		if sqrt((xs[i] - r36loc[1])^2 + (ys[i] - r36loc[2])^2) < r36loc[end]
			push!(r36infilnodes, i)
		end
	end
end
zonenums = [westzonenum; eastzonenum; wellzonenums; [336000]]
nodesinzones = [westnodes; eastnodes; wellzonenodes; [r36infilnodes]]
flownodes, skds, eflows, aipeds = FEHM.flattenzones(zonenums, nodesinzones, isanode, zoneornodenums, skds, eflows, aipeds)
dirichletnodes = Int[]
dirichletheads = Float64[]
for i = 1:length(flownodes)
	if skds[i] > 0
		push!(dirichletnodes, flownodes[i])
		push!(dirichletheads, skds[i])
	end
end
volumes, areasoverlengths, neighbors = FEHM.parsestor(joinpath(dirname(@__FILE__), "data/tet.stor"))
A, head, ch = FiniteVolume.solvediffusion(volumes, neighbors, areasoverlengths, ones(length(areasoverlengths)), zeros(length(volumes)), dirichletnodes, dirichletheads)
fig, ax = PyPlot.subplots()
ax[:plot](log10.(ch.data[:resnorm]))
display(fig); println()
PyPlot.close(fig)
nothing
