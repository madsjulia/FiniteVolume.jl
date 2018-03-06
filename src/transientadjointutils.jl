function getadjointfunctions(sigma, obsfreenodes, uobs, u0, tspan, Ss::Number, volumes::Vector, neighbors::Array{Pair{Int, Int}, 1}, areasoverlengths::Vector, conductivities::Vector, sources::Vector, dirichletnodes::Array{Int, 1}, dirichletheads::Vector, metaindex=i->i, logtransformconductivity=false; kwargs...)
	freenodes, nodei2freenodei = FiniteVolume.getfreenodes(length(u0), dirichletnodes)
	freenodei2nodei = Dict(zip(values(nodei2freenodei), keys(nodei2freenodei)))
	function g(u, t, uobs)
		uobseval = uobs(t)
		ueval = u(t)
		retval = 0.0
		for i in obsfreenodes
			retval += sigma(i, t)^2 * (ueval[freenodei2nodei[i]] - uobseval[freenodei2nodei[i]])^2
		end
		return retval
	end
	function dgdu(u, t, uobs)
		uobseval = uobs(t)
		ueval = u(t)
		result = zeros(sum(freenodes))
		for i in obsfreenodes
			result[i] = 2 * sigma(i, t)^2 * (ueval[freenodei2nodei[i]] - uobseval[freenodei2nodei[i]])
		end
		return result
	end
	function dfdp(u, t, p)
		ueval = u(t)[freenodes]
		p_conductivities = p[1:length(conductivities)]
		p_sources = p[length(conductivities) + 1:length(conductivities) + length(sources)]
		p_dirichletheads = p[length(conductivities) + length(sources) + 1:length(conductivities) + length(sources) + length(dirichletheads)]
		A_px = FiniteVolume.assembleA_px(ueval, neighbors, areasoverlengths, p_conductivities, p_sources, dirichletnodes, p_dirichletheads, metaindex, logtransformconductivity)
		b_p = FiniteVolume.assembleb_p(neighbors, areasoverlengths, p_conductivities, p_sources, dirichletnodes, p_dirichletheads, metaindex, logtransformconductivity)
		result = transpose(b_p - A_px)
		FiniteVolume.scalebyvolume!(result, Ss * volumes, freenodei2nodei)
		return transpose(result)
	end
	dgdpval = zeros(length(conductivities) + length(sources) + length(dirichletheads))
	function dgdp(u, t, p)
		return dgdpval
	end
	du0dp = spzeros(length(conductivities) + length(sources) + length(dirichletheads), sum(freenodes))
	function G(p)
		p_conductivities = p[1:length(conductivities)]
		p_sources = p[length(conductivities) + 1:length(conductivities) + length(sources)]
		p_dirichletheads = p[length(conductivities) + length(sources) + 1:length(conductivities) + length(sources) + length(dirichletheads)]
		us_p, ts_p = FiniteVolume.backwardeulerintegrate(u0, tspan, Ss, volumes, neighbors, areasoverlengths, p_conductivities, p_sources, dirichletnodes, p_dirichletheads, metaindex, logtransformconductivity; kwargs...)
		uc_p = FiniteVolume.getcontinuoussolution(us_p, ts_p)
		I, E = QuadGK.quadgk(t->g(uc_p, t, uobs), tspan...; maxevals=3 * 10^2, order=21)
		return I
	end
	return g, dgdu, dfdp, dgdp, du0dp, G
end
