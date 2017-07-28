using Base.Test
import FiniteVolume
import JLD
import PyPlot

neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads = JLD.load("waffledata.jld", "neighbors", "areasoverlengths", "hycos", "sources", "dirichletnodes", "dirichletheads")

#do the uniform hyco, no recharge case
@time head, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, ones(length(areasoverlengths)), zeros(length(sources)), dirichletnodes, dirichletheads)
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
@time hethead, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, hycos, zeros(length(sources)), dirichletnodes, dirichletheads)
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
@time fullhead, ch, A, b, freenode = FiniteVolume.solvediffusion(neighbors, areasoverlengths, hycos, sources, dirichletnodes, dirichletheads)
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
