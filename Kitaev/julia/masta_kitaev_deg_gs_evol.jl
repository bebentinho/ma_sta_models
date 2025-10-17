using Markdown
using InteractiveUtils
using LinearAlgebra
using QuantumOptics
using PyPlot

omega = 1 # Energy hopping
g_0 = 0
npart = Int(8)
tau = Int(120) # total time evolution
delta = 1
g_f = -3*omega

alpha_gvars = (64 * abs(delta)^2 * (sin(pi/npart))^2)^(-1)
beta_gvars = -2 * omega * cos(pi/npart)
gamma_gvars = 2 * abs(delta) * sin(pi/npart)

gvars = Dict("alpha" => alpha_gvars, "beta" => beta_gvars, "gamma" => gamma_gvars, "tau" => tau)
gvars_lin = Dict("gf" => g_f, "tau" => tau)

# Formatting parameters
amt_times = 200 # How many points we calculate
ts = LinRange(0, tau, amt_times)

function g_t(t, gvars)
  # Minimal action solution
  gvars["beta"] + gvars["gamma"] * tan((1-t/gvars["tau"]) * atan((g_0 - gvars["beta"])/gvars["gamma"]) + t/gvars["tau"] * atan((g_f - gvars["beta"])/gvars["gamma"]) )
end

function g_lin_t(t, gvars)
    # Linear ramp solution
    g_0 + (gvars["gf"] - g_0) * t / gvars["tau"]
end

id = identityoperator(SpinBasis(1/2))
σˣ = sigmax(SpinBasis(1/2))
σʸ = sigmay(SpinBasis(1/2))
σᶻ = sigmaz(SpinBasis(1/2))

function Hstatic(N)
	# vector of operators: [σˣ, σˣ, id, ...]
    🍎 = [σˣ; σˣ; fill(id, N-2)]
    
    # vector of operators: [σˣ, id, ...]
    🍉 = [σʸ; σʸ; fill(id, N-2)]
    
    H = 0*tensor(fill(id, N)...)
    for i in 1:N-1
        # tensor multiply all operators
        H -= 0.5 * (omega+delta) * tensor(🍎...)
		H -= 0.5 * (omega-delta) * tensor(🍉...)
        # cyclic shift the operators
        🍎 = circshift(🍎,1)
		🍉 = circshift(🍉,1)
    end
	
	H
end

function Hdyn(N)
	# vector of operators
	🥭 = [σᶻ; fill(id, N-1)]
	
	H = 0*tensor(fill(id, N)...)
    for i in 1:N
        # tensor multiply all operators
        H -= 0.5 * tensor(🥭...)
        # cyclic shift the operators
        🥭 = circshift(🥭,1)
    end

	H
end

function H_t(t, gvars, N)
	Hstatic(N) + g_t(t, gvars)*Hdyn(N)
end

function H_lin_t(t, gvars, N)
	Hstatic(N) + g_lin_t(t, gvars)*Hdyn(N)
end

# Now let's calculate the fidelity

tmin = 0.5
tmax = 120

fidel_interval = vcat(LinRange(tmin,tmax,amt_times))

eigens_0, eigsts_0 = eigenstates(dense(H_t(0, gvars, npart)), 2)
gs1_0 = eigsts_0[1]
gs2_0 = eigsts_0[2]

eigenslin_0, eigstslin_0 = eigenstates(dense(H_lin_t(0, gvars_lin, npart)), 2)
gslin1_0 = eigstslin_0[1]
gslin2_0 = eigstslin_0[2]

fidel1 = zeros(amt_times)
fidel2 = zeros(amt_times)

fidel_lin1 = zeros(amt_times)
fidel_lin2 = zeros(amt_times)

for index in 1:amt_times
	
	time = fidel_interval[index]
	
	gvarsaux = Dict("alpha" => alpha_gvars, "beta" => beta_gvars, "gamma" => gamma_gvars, "tau" => time)
	gvars_linaux = Dict("gf" => g_f, "tau" => time)
	
	eigens_t, eigsts_t = eigenstates(dense(H_t(time, gvarsaux, npart)), 2)
	gs_t = eigsts_t[1]
	es_t = eigsts_t[2]
	
	eigenslin_t, eigstslin_t = eigenstates(dense(H_lin_t(time, gvars_linaux, npart)), 2)
	gs_lin_t = eigstslin_t[1]
	es_lin_t = eigstslin_t[2]
	
	evol1 = timeevolution.schroedinger_dynamic([0,time], gs1_0, (t, psi) -> H_t(t, gvarsaux, npart))
	evol2 = timeevolution.schroedinger_dynamic([0,time], gs2_0, (t, psi) -> H_t(t, gvarsaux, npart))
	
	evol_lin1 = timeevolution.schroedinger_dynamic([0,time], gslin1_0, (t, psi) -> H_lin_t(t, gvars_linaux, npart))
	evol_lin2 = timeevolution.schroedinger_dynamic([0,time], gslin2_0, (t, psi) -> H_lin_t(t, gvars_linaux, npart))
	
	fidel1[index] = norm(dagger(gs_t)*evol1[2][2])^2
	fidel2[index] = norm(dagger(gs_t)*evol2[2][2])^2
	fidel_lin1[index] = norm(dagger(gs_lin_t)*evol_lin1[2][2])^2
	fidel_lin2[index] = norm(dagger(gs_lin_t)*evol_lin2[2][2])^2
end

begin
    # Plot the results
    fig, ax = plt.subplots()

    # Plot the results for linear ramp and action ramp
    ax.plot(fidel_interval, fidel2, color="firebrick", linestyle="-.", label="MA even")
    ax.plot(fidel_interval, fidel_lin2, color="blue", linestyle="-.", label="LR even")
    ax.plot(fidel_interval, fidel1, color="firebrick", label="MA odd")
    ax.plot(fidel_interval, fidel_lin1, color="blue", label="LR odd")
    
    # Add labels and grid
    ax.set_xlabel(r"$\omega \tau$")
    ax.set_ylabel(r"$\mathcal{F}$")
    ax.legend()
    ax.grid()

    # Show the plot
    plt.xlim(0, tmax)

    plt.ylim(0, 1.0)

    gcf() # Get current figure to display in Pluto
end
