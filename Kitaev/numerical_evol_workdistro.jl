# using Markdown
# using InteractiveUtils
using LinearAlgebra
using QuantumOptics
using CairoMakie
using ProgressMeter
using GeometryBasics
using JLD2
using DelimitedFiles


omega = 1 # Energy hopping
g_0 = 0
npart = Int(14)
tau = Int(30) # total time evolution
delta = omega
g_f = -3*omega

alpha_gvars = (64 * abs(delta)^2 * (sin(pi/npart))^2)^(-1)
beta_gvars = -2 * omega * cos(pi/npart)
gamma_gvars = 2 * abs(delta) * sin(pi/npart)

gvars = Dict("alpha" => alpha_gvars, "beta" => beta_gvars, "gamma" => gamma_gvars, "tau" => tau)
gvars_lin = Dict("gf" => g_f, "tau" => tau)

# Formatting parameters

amt_times = 200 # How many points we calculate
ts = LinRange(0, tau, amt_times)

id = identityoperator(SpinBasis(1/2))
σˣ = sigmax(SpinBasis(1/2))
σʸ = sigmay(SpinBasis(1/2))
σᶻ = sigmaz(SpinBasis(1/2))

function g_t(t, gvars)
  # Minimal action solution
  gvars["beta"] + gvars["gamma"] * tan((1-t/gvars["tau"]) * atan((g_0 - gvars["beta"])/gvars["gamma"]) + t/gvars["tau"] * atan((g_f - gvars["beta"])/gvars["gamma"]) )
end

function g_lin_t(t, gvars)
    # Linear ramp solution
    g_0 + (gvars["gf"] - g_0) * t / gvars["tau"]
end

function ℙₑ(state)
	round(1 + expect(tensor(fill(σᶻ, npart)...), state))/2
end

function ℙₒ(state)
	round(1 - expect(tensor(fill(σᶻ, npart)...), state))/2
end

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


function calculate_work_distribution(H_final, evolved_state, E_initial)

    evals_f, estates_f = eigenstates(dense(H_final))
    
    work_values = evals_f .- E_initial
    probabilities = [abs2(dagger(estates_f[n]) * evolved_state) for n in 1:length(evals_f)]
    
    return work_values, probabilities
end

function plot_work_evolution_3d(all_tau_data, title_str, filename)
        
    set_theme!(theme_latexfonts())
    fig = Figure(resolution = (1000, 800))
    ax = Axis3(fig[1, 1],
        title = title_str,
        xlabel = L"Work $W$",
        ylabel = L"Total Time $\tau$",
        zlabel = L"Probability $P(W)$",
        perspectiveness = 0.5,
        azimuth = 1.2, # Ângulo de visão
        elevation = 0.3
    )

    # Cores para diferenciar os taus (degradê)
    colors = cgrad(:viridis, length(all_tau_data), categorical = true)

    for (i, data) in enumerate(all_tau_data)
        τ_val, W, P = data
        
        # Criamos uma linha vertical para cada nível de energia no plano 3D
        # O eixo Y agora representa o valor de Tau
        for (w_val, p_val) in zip(W, P)
            lines!(ax, [w_val, w_val], [τ_val, τ_val], [0, p_val], 
                color = colors[i], linewidth = 2)
        end
        
        # Opcional: Uma linha no "chão" conectando os picos para guiar o olho
        scatter!(ax, W, fill(τ_val, length(W)), P, 
                color = colors[i], markersize = 5)
    end

    save(filename, fig)
    return fig
end


# Now let's calculate the fidelity

tmin = 0.5
tmax = tau

fidel_interval = vcat(LinRange(tmin,tmax,amt_times))

# eigens_0, eigsts_0 = eigenstates(dense(H_t(0, gvars, npart)), 2)
eigen_0 = -13.000000000000002
# gs1_0 = eigsts_0[1]
# gs2_0 = eigsts_0[2]

# gse_0 = ℙₑ(gs1_0)*gs1_0 + ℙₑ(gs2_0)*gs2_0
# gso_0 = ℙₒ(gs1_0)*gs1_0 + ℙₒ(gs2_0)*gs2_0


# eigenslin_0, eigstslin_0 = eigenstates(dense(H_lin_t(0, gvars_lin, npart)), 2)
# gslin1_0 = eigstslin_0[1]
# gslin2_0 = eigstslin_0[2]

# gsline_0 = ℙₑ(gslin1_0)*gslin1_0 + ℙₑ(gslin2_0)*gslin2_0
# gslino_0 = ℙₒ(gslin1_0)*gslin1_0 + ℙₒ(gslin2_0)*gslin2_0


# fidele = zeros(amt_times)
# fidelo = zeros(amt_times)
# fidel_line = zeros(amt_times)
# fidel_lino = zeros(amt_times)

# array_evole = []
# array_evolo = []
# array_evol_line = []
# array_evol_lino = []

# @showprogress for index in 1:amt_times
# 	time = fidel_interval[index]
	
# 	gvarsaux = Dict("alpha" => alpha_gvars, "beta" => beta_gvars, "gamma" => gamma_gvars, "tau" => time)
# 	gvars_linaux = Dict("gf" => g_f, "tau" => time)
	
# 	eigens_t, eigsts_t = eigenstates(H_t(time, gvarsaux, npart), 2, info=false)
# 	gs_t = eigsts_t[1]
# 	es_t = eigsts_t[2]
	
# 	eigenslin_t, eigstslin_t = eigenstates(H_lin_t(time, gvars_linaux, npart), 2, info=false)
# 	gs_lin_t = eigstslin_t[1]
# 	es_lin_t = eigstslin_t[2]
	
# 	evole = timeevolution.schroedinger_dynamic([0,time], gse_0, (t, psi) -> H_t(t, gvarsaux, npart))
# 	evolo = timeevolution.schroedinger_dynamic([0,time], gso_0, (t, psi) -> H_t(t, gvarsaux, npart))
	
# 	evol_line = timeevolution.schroedinger_dynamic([0,time], gsline_0, (t, psi) -> H_lin_t(t, gvars_linaux, npart))
# 	evol_lino = timeevolution.schroedinger_dynamic([0,time], gslino_0, (t, psi) -> H_lin_t(t, gvars_linaux, npart))

#     push!(array_evolo, evolo[2][2])
#     push!(array_evole, evole[2][2])
#     push!(array_evol_lino, evol_lino[2][2])
#     push!(array_evol_line, evol_line[2][2])

# 	fidelo[index] = norm(dagger(es_t)*evolo[2][2])^2
# 	fidele[index] = norm(dagger(gs_t)*evole[2][2])^2
# 	fidel_lino[index] = norm(dagger(es_lin_t)*evol_lino[2][2])^2
# 	fidel_line[index] = norm(dagger(gs_lin_t)*evol_line[2][2])^2
# end

basis = tensor(fill(SpinBasis(1/2), npart)...)

@load "matriz_evolo_indexes.jld2" matriz_evolo_indexes
@load "matriz_evole_indexes.jld2" matriz_evole_indexes
# @load "matriz_evol_line.jld2" matriz_evol_line
# @load "matriz_evol_lino.jld2" matriz_evol_lino


array_evolo = [Ket(basis, matriz_evolo_indexes[:, i]) for i in 1:size(matriz_evolo_indexes, 2)]
array_evole = [Ket(basis, matriz_evole_indexes[:, i]) for i in 1:size(matriz_evole_indexes, 2)]
# array_evol_line = [Ket(basis, matriz_evol_line[:, i]) for i in 1:size(matriz_evol_line, 2)]
# array_evol_lino = [Ket(basis, matriz_evol_lino[:, i]) for i in 1:size(matriz_evol_lino, 2)]


# matriz_evolo = hcat([k.data for k in array_evolo]...)
# @save "matriz_evolo.jld2" matriz_evolo
# matriz_evole = hcat([k.data for k in array_evole]...)
# @save "matriz_evole.jld2" matriz_evole
# matriz_evol_line = hcat([k.data for k in array_evol_line]...)
# @save "matriz_evol_line.jld2" matriz_evol_line
# matriz_evol_lino = hcat([k.data for k in array_evol_lino]...)
# @save "matriz_evol_lino.jld2" matriz_evol_lino


# writedlm("ma_sta_stuff/array_evolo.txt", array_evolo)
# writedlm("ma_sta_stuff/array_evole.txt", array_evole)
# writedlm("ma_sta_stuff/array_evol_line.txt", array_evol_line)
# writedlm("ma_sta_stuff/array_evol_lino.txt", array_evol_lino)

# writedlm("ma_sta_stuff/fidelo.txt", fidelo)
# writedlm("ma_sta_stuff/fidele.txt", fidele)
# writedlm("ma_sta_stuff/fidel_line.txt", fidel_line)
# writedlm("ma_sta_stuff/fidel_lino.txt", fidel_lino)

index_to_plot = [Int(amt_times/10):Int(amt_times/10):amt_times;]
# index_to_plot = [amt_times/2]
data_3d_even_MA = []
data_3d_odd_MA = []
# data_3d_even_LR = []
# data_3d_odd_LR = []

tau_list = []
W_even_list = []
P_even_list = []
W_odd_list = []
P_odd_list = []

@showprogress for index in index_to_plot

    τ = fidel_interval[index]
    Hf_MA = H_t(τ, gvars, npart)
    # Hf_LR = H_lin_t(τ, gvars_lin, npart)


    W_even_MA, P_even_MA = calculate_work_distribution(Hf_MA, array_evole[Int(index*10/amt_times)], eigen_0)
    W_odd_MA, P_odd_MA = calculate_work_distribution(Hf_MA, array_evolo[Int(index*10/amt_times)], eigen_0)
    # W_even_LR, P_even_LR = calculate_work_distribution(Hf_LR, array_evol_line[Int(index*10/amt_times)], eigens_0[1])
    # W_odd_LR, P_odd_LR = calculate_work_distribution(Hf_LR, array_evol_lino[Int(index*10/amt_times)], eigens_0[1])

    push!(tau_list, τ)
    push!(W_even_list, W_even_MA)
    push!(P_even_list, P_even_MA)
    push!(W_odd_list, W_odd_MA)
    push!(P_odd_list, P_odd_MA)

    push!(data_3d_even_MA, (τ, W_even_MA, P_even_MA))
    push!(data_3d_odd_MA, (τ, W_odd_MA, P_odd_MA))
    # push!(data_3d_even_LR, (τ, W_even_LR, P_even_LR))
    # push!(data_3d_odd_LR, (τ, W_odd_LR, P_odd_LR))

end

writedlm("tau_list.txt", tau_list)
writedlm("W_even_list.txt", W_even_list)
writedlm("P_even_list.txt", P_even_list)
writedlm("W_odd_list.txt", W_odd_list)
writedlm("P_odd_list.txt", P_odd_list)

function plot_work_evolution_3d_volumetric(all_tau_data, filename, colormap = :viridis)
    set_theme!(fonts = (; regular = "Computer Modern Roman", bold = "Computer Modern Bold", italic = "Computer Modern Italic", bold_italic = "Computer Modern Bold Italic"), fontsize = 20)
    fig = Figure(size = (1000, 800))
    ax = Axis3(fig[1, 1],
        xlabel = L"$W/\omega$",
        ylabel = L"$\omega \tau$",
        zlabel = L"$P(W/\omega)$",
        azimuth = 0.25 * π,
        elevation = 0.1 * π,
        limits = ((-10,10),(0.0, tau+5.0), (0,1)),
        xgridvisible = true, ygridvisible = true, zgridvisible = true,
        protrusions = (100, 100, 100, 100) 
    )

    colors = cgrad(colormap, length(all_tau_data), categorical = true)
    for (i, data) in enumerate(all_tau_data)
        τ_val, W, P = data
        
        # scatter!(ax, W, fill(τ_val, length(W)), P, 
                # color = colors[i], markersize = 5)

        # Criando "cilindros" em vez de linhas finas
        # O Rect3f cria uma caixa (prisma) que é mais visível
        for (w_val, p_val) in zip(W, P)
            
            largura = 0.3
            profundidade = 0.01
            alpha = 0.6
            if p_val > 0.001
                mesh!(ax, 
                        Rect3f(Vec3f(w_val - largura/2, τ_val - profundidade/2, 0), # Origem
                            Vec3f(largura, profundidade, p_val)),              # Dimensões (largura, profundidade, altura)
                        color = (colors[i], alpha),
                        shading = false
                )
            end    
        end


    end

    save(filename, fig)
    return fig
end



fig = plot_work_evolution_3d_volumetric(data_3d_even_MA, "pw_3d_even_MA.png", :viridis)
fig = plot_work_evolution_3d_volumetric(data_3d_odd_MA, "pw_3d_odd_MA.png", :plasma)
# fig = plot_work_evolution_3d_volumetric(data_3d_even_LR, "pw_3d_even_LR.png", :viridis)
# fig = plot_work_evolution_3d_volumetric(data_3d_odd_LR, "pw_3d_odd_LR.png", :plasma)