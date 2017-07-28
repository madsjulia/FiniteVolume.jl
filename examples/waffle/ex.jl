using Base.Test
import FEHM
import FiniteVolume
import PyPlot

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
if !isdefined(:volumes)
	volumes, areasoverlengths, neighbors = FEHM.parsestor(joinpath(dirname(@__FILE__), "data/tet.stor"))
end
hycos = FiniteVolume.fehmhyco2fvhyco(xs, ys, zs, kxs, kys, kzs, neighbors)
#do the uniform hyco, no recharge case
@time head, ch, A, b, freenode = FiniteVolume.solvediffusion(volumes, neighbors, areasoverlengths, ones(length(areasoverlengths)), zeros(length(volumes)), dirichletnodes, dirichletheads)
fehmhead = readdlm("data/w01.00007_sca_node.avs.hom_norecharge"; skipstart=2)[:, 2]
@show norm(A * head[freenode] - b)
@show norm(A * fehmhead[freenode] - b)
@show norm(b)
@show norm(head - fehmhead)
@show mean(head - fehmhead)
@show mean(abs.(head - fehmhead))
fig, ax = PyPlot.subplots()
ax[:plot](log10.(ch.data[:resnorm]))
display(fig); println()
PyPlot.close(fig)

#do the heterogeneous hyco, no recharge case
@time hethead, ch, A, b, freenode = FiniteVolume.solvediffusion(volumes, neighbors, areasoverlengths, hycos, zeros(length(volumes)), dirichletnodes, dirichletheads)
fehmhethead = readdlm("data/w01.00007_sca_node.avs.het_norecharge"; skipstart=2)[:, 2]
@show norm(A * hethead[freenode] - b)
@show norm(A * fehmhethead[freenode] - b)
@show norm(b)
@show norm(hethead - fehmhethead)
@show mean(hethead - fehmhethead)
@show mean(abs.(hethead - fehmhethead))
fig, ax = PyPlot.subplots()
ax[:plot](log10.(ch.data[:resnorm]))
display(fig); println()
PyPlot.close(fig)

#do the heterogeneous hyco, recharge case
@time fullhead, ch, A, b, freenode = FiniteVolume.solvediffusion(volumes, neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
fehmfullhead = readdlm("data/w01.00007_sca_node.avs.het_recharge"; skipstart=2)[:, 2]
@show norm(A * fullhead[freenode] - b)
@show norm(A * fehmfullhead[freenode] - b)
@show norm(b)
@show norm(fullhead - fehmfullhead)
@show mean(fullhead - fehmfullhead)
@show mean(abs.(fullhead - fehmfullhead))
fig, ax = PyPlot.subplots()
ax[:plot](log10.(ch.data[:resnorm]))
display(fig); println()
PyPlot.close(fig)

nothing
