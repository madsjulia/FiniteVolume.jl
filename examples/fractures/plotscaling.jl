import PyPlot

numnodes = Dict()
numnodes["circuit"] = 6280
numnodes["backbone_x01"] = 360912
numnodes["homogenous-10m"] = 7758411
numnodes["pl_alpha_1.6"] = 9384128
numnodes["25L_network_x2"] = 18663887

mbploadtime = Dict()
mbpsolvetime = Dict()
mbploadtime["circuit"] = 0.006093
mbpsolvetime["circuit"] = 0.085352
mbploadtime["backbone_x01"] = 0.256931
mbpsolvetime["backbone_x01"] = 1.702068
mbploadtime["homogenous-10m"] = 3.145018
mbpsolvetime["homogenous-10m"] = 143.332097
mbploadtime["pl_alpha_1.6"] = 3.697401
mbpsolvetime["pl_alpha_1.6"] = 174.389309
mbploadtime["25L_network_x2"] = 8.194946
mbpsolvetime["25L_network_x2"] = 129.239627

mmloadtime = Dict()
mmsolvetime = Dict()
mmloadtime["circuit"] = 0.087875
mmsolvetime["circuit"] = 0.277772
mmloadtime["backbone_x01"] = 1.353354
mmsolvetime["backbone_x01"] = 2.402553
mmloadtime["homogenous-10m"] = 18.866763
mmsolvetime["homogenous-10m"] = 265.082275
mmloadtime["pl_alpha_1.6"] = 23.845679
mmsolvetime["pl_alpha_1.6"] = 317.173328
mmloadtime["25L_network_x2"] = 59.654626
mmsolvetime["25L_network_x2"] = 241.924559

es1loadtime = Dict()
es1solvetime = Dict()
es1loadtime["circuit"] = 0.108505
es1solvetime["circuit"] = 0.155559
es1loadtime["backbone_x01"] = 0.902550
es1solvetime["backbone_x01"] = 1.781102
es1loadtime["homogenous-10m"] = 16.414932
es1solvetime["homogenous-10m"] = 157.650631
es1loadtime["pl_alpha_1.6"] = 19.951799
es1solvetime["pl_alpha_1.6"] = 178.184294
es1loadtime["25L_network_x2"] = 40.877704
es1solvetime["25L_network_x2"] = 130.685777

pflotrantime = Dict()
pflotrancores = Dict()
pflotrantime["circuit"] = 1.0737E+00
pflotrancores["circuit"] = 1
pflotrantime["backbone_x01"] = 2.3955E+01
pflotrancores["backbone_x01"] = 32
pflotrantime["homogenous-10m"] = 1.2871E+03
pflotrancores["homogenous-10m"] = 16
pflotrantime["pl_alpha_1.6"] = 9.8432E+02
pflotrancores["pl_alpha_1.6"] = 32
pflotrantime["25L_network_x2"] = 3.0245E+03
pflotrancores["25L_network_x2"] = 1

fig, axs = PyPlot.subplots(1, 2, figsize=(12, 4.5))
for x in ["circuit", "backbone_x01", "homogenous-10m", "pl_alpha_1.6", "25L_network_x2"]
	for i = 1:2
		axs[i][:loglog](numnodes[x], mbploadtime[x] + mbpsolvetime[x], "b.", ms=10, alpha=0.5)
	end
	for i = 1:2
		axs[i][:loglog](numnodes[x], mmloadtime[x] + mmsolvetime[x], "r.", ms=10, alpha=0.5)
	end
	for i = 1:2
		axs[i][:loglog](numnodes[x], es1loadtime[x] + es1solvetime[x], "g.", ms=10, alpha=0.5)
	end
	axs[1][:loglog](numnodes[x], pflotrantime[x], "k.", ms=10, alpha=0.5)
	axs[2][:loglog](numnodes[x], pflotrantime[x] * pflotrancores[x], "k.", ms=10, alpha=0.5)
	@show x
	@show (pflotrantime[x]) / (mbploadtime[x] + mbpsolvetime[x])
	@show (pflotrantime[x] * pflotrancores[x]) / (mbploadtime[x] + mbpsolvetime[x])
end
for i = 1:2
	axs[i][:set_xlabel]("number of nodes")
end
axs[1][:legend](["FV.jl (laptop)", "FV.jl (madsmax)",  "FV.jl (es01)","PFLOTRAN"], loc=2)
axs[1][:set_ylabel]("Wall time [s]")
axs[2][:legend](["FV.jl (laptop)", "FV.jl (madsmax)",  "FV.jl (es01)","PFLOTRAN"], loc=2)
axs[2][:set_ylabel]("CPU time [s]")
fig[:tight_layout]
fig[:savefig]("scaling.pdf")
display(fig)
println()
PyPlot.close(fig)
