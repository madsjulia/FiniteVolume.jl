import FEHM
import FiniteVolume

westzonenum, westnodes = FEHM.parsezone(joinpath(dirname(@__FILE__), "data/out_west.zonn"))
eastzonenum, eastnodes = FEHM.parsezone(joinpath(dirname(@__FILE__), "data/out_east.zonn"))
wellzonenums, wellzonenodes = FEHM.parsezone(joinpath(dirname(@__FILE__), "data/well_screens.zonn"))
isanode, zoneornodenums, skds, eflows, aipeds = FEHM.parseflow(joinpath(dirname(@__FILE__), "data/wl.flow"))
hycoisanode, hycozoneornodenums, kxs, kys, kzs = FEHM.parsehyco(joinpath(dirname(@__FILE__), "data/w01.hyco"))
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
hyconodes, kxs, kys, kzs = FEHM.flattenzones(zonenums, nodesinzones, hycoisanode, hycozoneornodenums, kxs, kys, kzs)#hyco is in m/s
dirichletnodes = Int[]
dirichletheads = Float64[]
sources = zeros(length(xs))
for i = 1:length(flownodes)
	if skds[i] > 0
		push!(dirichletnodes, flownodes[i])
		push!(dirichletheads, skds[i])
	else
		#this means skds[i] is a mass source in units of kg/s
		sources[flownodes[i]] = -skds[i] * 1e-3#convert of kg/s to m^3/s
	end
end
volumes, areasoverlengths, neighbors = FEHM.parsestor(joinpath(dirname(@__FILE__), "data/tet.stor"))
goodindices = filter(i->neighbors[i][1] < neighbors[i][2], 1:length(neighbors))
areasoverlengths = areasoverlengths[goodindices]
neighbors = neighbors[goodindices]
hycos = FiniteVolume.fehmhyco2fvhyco(xs, ys, zs, kxs, kys, kzs, neighbors)

JLD.save("waffledata.jld", "neighbors", neighbors, "areasoverlengths", areasoverlengths, "hycos", hycos, "sources", sources, "dirichletnodes", dirichletnodes, "dirichletheads", dirichletheads)
