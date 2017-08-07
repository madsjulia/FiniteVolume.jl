import FEHM
import JLD

coords = FEHM.parsegrid(joinpath(dirname(@__FILE__), "data", "wtr25.fehmn"))
xs = coords[1, :]
ys = coords[2, :]
zs = coords[3, :]
volumes, areasoverlengths, neighbors = FEHM.parsestor(joinpath(dirname(@__FILE__), "data", "wtr25.stor"))
goodindices = filter(i->neighbors[i][1] < neighbors[i][2], 1:length(neighbors))
areasoverlengths = areasoverlengths[goodindices]
neighbors = neighbors[goodindices]
zonenums, nodesinzones = FEHM.parsezone(joinpath(dirname(@__FILE__), "data", "wtr25_outside.zone"))
#zones are 1=top, 2=bottom, 3=left/west, 5=right/east, 6=back/north, 4=front/south
dirichletnodes = unique(vcat(nodesinzones[[3, 4, 5, 6]]...))
rechargenodes = collect(setdiff(nodesinzones[1], dirichletnodes))
topnodes = nodesinzones[1]
hycoxs = Float64[]
hycoys = Float64[]
hycozs = Float64[]
topnodeset = Set(topnodes)
tophyconodes = Int[]
for (i, (node1, node2)) in enumerate(neighbors)
	if node1 in topnodeset && node2 in topnodeset
		push!(tophyconodes, i)
	end
	push!(hycoxs, .5 * (xs[node1] + xs[node2]))
	push!(hycoys, .5 * (ys[node1] + ys[node2]))
	push!(hycozs, .5 * (zs[node1] + zs[node2]))
end

JLD.save("model.jld", "neighbors", neighbors, "areasoverlengths", areasoverlengths, "dirichletnodes", dirichletnodes, "rechargenodes", rechargenodes, "xs", xs, "ys", ys, "zs", zs, "topnodes", topnodes, "hycoxs", hycoxs, "hycoys", hycoys, "hycozs", hycozs, "tophyconodes", tophyconodes)
