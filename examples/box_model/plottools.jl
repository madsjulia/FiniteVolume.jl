import PyPlot

function plotlayers(layers, fieldsasvector...)
	fig, axs = PyPlot.subplots(length(fieldsasvector), length(layers), figsize=(16, 9))
	axs = reshape(axs, length(fieldsasvector), length(layers))
	high = max(map(maximum, fieldsasvector)...)
	low = min(map(minimum, fieldsasvector)...)
	for i = 1:length(fieldsasvector)
		field = reshape(fieldsasvector[i], ns[3], ns[2], ns[1])
		for j = 1:length(layers)
			axs[i, j][:imshow](Matrix(field[layers[j], :, :]), vmin=low, vmax=high, interpolation="nearest")
		end
	end
	display(fig)
	println()
	PyPlot.close(fig)
end
