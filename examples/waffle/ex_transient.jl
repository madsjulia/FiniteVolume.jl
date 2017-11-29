using Base.Test
import FiniteVolume
import JLD
import LinearAdjoints
import Optim
import PyPlot
import ReusableFunctions
import CrPlots

if !isdefined(:fullhead)
	@time neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads, xs, ys, zs, topnodes = JLD.load("waffledata.jld", "neighbors", "areasoverlengths", "hycos", "sources", "dirichletnodes", "dirichletheads", "xs", "ys", "zs", "topnodes")

	#do the heterogeneous hyco, recharge case
	@time fullhead, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
	fehmfullhead = readdlm("data/w01.00007_sca_node.avs.het_recharge"; skipstart=2)[:, 2]
	@test norm(A * fehmfullhead[freenode] - b) / norm(b) < 1e-3
end

r28nodes = [731459, 714043]
sources[r28nodes] = -1.e-3#this is in m^3/s and translates to ~15gpm
print("transient")
if !isdefined(:us)
	@time us, ts = FiniteVolume.backwardeulerintegrate(fullhead, (0.0, 60*60*24*365*5), neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
end
fig, ax = PyPlot.subplots()
ax[:plot](ts, map(u->u[731459], us))
display(fig); println()
PyPlot.close(fig)

r28inithead = us[1][731459]
r28finalhead = us[end][731459]
wells = ["R-01", "R-11", "R-13", "R-15", "R-28", "R-33", "R-34", "R-35b", "R-36", "R-42", "R-43", "R-44", "R-45", "R-50", "R-61", "R-62", "CrEX-1", "CrEX-3", "CrIN-1", "CrIN-2", "CrIN-3", "CrIN-4", "CrIN-5"]
boundingbox = (497291.54, 537425.97, 501312.87, 539628.56)
for i = 1:length(us)
	fig, ax, img = CrPlots.crplot(boundingbox, xs[topnodes[1]], ys[topnodes[1]], us[i][topnodes[1]]; alpha=0.5)
	CrPlots.addwells(ax, wells)
	CrPlots.addcbar(fig, img, "Head (m)", CrPlots.getticks([r28finalhead, r28inithead]), r28finalhead, r28inithead)
	CrPlots.addmeter(ax, boundingbox[1] + 500, boundingbox[2] + 250, [250, 500, 1000], ["250m", "500m", "1km"])
	CrPlots.addpbar(fig, ax, (ts[i] - minimum(ts)) / (maximum(ts) - minimum(ts)), "5 Years")
	fig[:savefig]("figs/wl_r28pumping_$(lpad(i, 4, 0)).png")
	display(fig)
	PyPlot.close(fig)
	println()
end
