using Base.Test
import Interpolations

#this test is based on the closed form example from https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf
#here we compute the gradient of ∫xdt where dx/dt=bx and x(0)=a
#the analytical solution is x(t)=a*exp(b*t)
#the adjoint solution is λ(t)=(1-exp(b*(T-t)))/b
#the gradient is a / b * (exp(b*T) - 1)
#the parameters are p = [a, b]
a = 1.0
b = 2.0
p = [a, b]
T = 1.0
x(t, p) = [p[1] * exp(p[2] * t)]
lambda(t, p) = (1 - exp(p[2] * (T - t))) / b
G(p) = p[1] / p[2] * (exp(p[2] * T) - 1)
gradient(p) = [1 / p[2] * (exp(p[2] * T) - 1), -p[1] / p[2]^2 * (exp(p[2] * T) - 1) + p[1] / p[2] * exp(p[2] * T) * T]
A = p->fill(p[2], 1, 1)
getb(t, p) = zeros(1)
function dx0dp(p)
	result = Array{Float64}(2, 1)
	result[1, 1] = -1.0
	result[2, 1] = 0
	return result
end
function dfdp(xc, t)
	result = zeros(2, 1)
	result[2, 1] = -xc(t)[1]
	return result
end
getdgdu(t) = -ones(1)
linearsolver(A, b, x0) = A \ b
x0 = x(0, p)
xs, ts_x = FiniteVolume.backwardeulerintegrate(x0, -A(p), t->getb(t, p), 1e-5, 0.0, T; linearsolver=linearsolver, atol=1e-8)
@test isapprox(xs, map(t->x(t, p), ts_x), rtol=1e-4)
lambdas, ts_lambda = FiniteVolume.adjointintegrate(-A(p), getdgdu, (0.0, T); dt0=1e-5, linearsolver=linearsolver, atol=1e-8)
@test isapprox(lambdas, map(t->lambda(t, p), ts_lambda), rtol=1e-4)
xc = FiniteVolume.getcontinuoussolution(xs, ts_x)
lambdac = FiniteVolume.getcontinuoussolution(lambdas, ts_lambda)
adjointgradient, E = FiniteVolume.gradientintegrate(lambdac, dx0dp(p), t->[0, 0], t->dfdp(xc, t), (0.0, T))
@test isapprox(adjointgradient, gradient(p); rtol=1e-4)
