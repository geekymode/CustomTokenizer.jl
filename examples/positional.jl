# Positional encoding: what each scheme does, checked rather than asserted.
#
#   julia --project=docs examples/positional.jl

using CustomTokenizer, CairoMakie, Printf, Random, LinearAlgebra

CairoMakie.activate!(type = "png")
figs = joinpath(@__DIR__, "figures"); mkpath(figs)

section(t) = println("\n", "="^72, "\n", t, "\n", "="^72)

section("why anything is needed at all")
println("""
Attention scores are dot products between vectors, so reordering the input just
reorders the output. Without position, "the cat chases the mouse" and "the mouse
chases the cat" are the same bag of vectors. Position has to be added by hand.""")

section("1. sinusoidal — fixed, added to the token vectors")
P = sinusoidal_encoding(8, 6)
println("dimension × position, first six positions of an 8-dimensional encoding:\n")
for r in 1:8
    @printf("  row %d  %s\n", r, join((@sprintf("%+6.3f", P[r, p]) for p in 1:6), " "))
end
println("""
Rows come in pairs, each pair a clock: sin and cos of the same angle. The top
pair ticks every few positions, the bottom pair over the whole context.""")

Plong = sinusoidal_encoding(8, 32)
k = 3
θ = k / 10_000.0^(2 * 1 / 8)          # the second clock, which turns slowly enough to read
println("\n  PE[:, p+$k] is PE[:, p] turned by one fixed angle, the same at every p:")
for p in (0, 5, 17)
    sinp, cosp = Plong[3, p + 1], Plong[4, p + 1]
    @printf("    p=%2d: (%+.4f, %+.4f) → rotated (%+.4f, %+.4f) · actual (%+.4f, %+.4f)\n",
            p, sinp, cosp,
            sinp * cos(θ) + cosp * sin(θ), cosp * cos(θ) - sinp * sin(θ),
            Plong[3, p + k + 1], Plong[4, p + k + 1])
end

section("2. learned table — one column per position, trained")
L = learned_positions(8, 512)
@printf("a %d × %d table of parameters; position 512 is the last one that exists,\n", size(L)...)
println("so the model cannot run beyond the length it was trained for.")

section("3. RoPE — rotate the query and key instead of adding")
rng = Xoshiro(1)
q, k2 = randn(rng, 16), randn(rng, 16)
@printf("  rotation preserves length:      |q| = %.4f, |rope(q, 137)| = %.4f\n",
        norm(q), norm(rope(q, 137)))
@printf("  position 0 does nothing:        %s\n", rope(q, 0) ≈ q)
println("\n  only the gap matters — absolute positions cancel:")
for (m, n) in ((7, 3), (100, 96), (2, 9))
    lhs = dot(rope(q, m), rope(k2, n))
    rhs = dot(rope(q, m - n), k2)
    @printf("    ⟨rope(q,%3d), rope(k,%3d)⟩ = %+10.6f   ⟨rope(q,%3d), k⟩ = %+10.6f   %s\n",
            m, n, lhs, m - n, rhs, isapprox(lhs, rhs) ? "same" : "DIFFERENT")
end
println("\n  that is why RoPE is applied inside attention, to q and k, at every layer.")

section("4. ALiBi — no vectors at all, a penalty on the score")
for (h, m) in enumerate(alibi_slopes(4))
    @printf("  head %d, slope %.4f: a token 10 away loses %.2f, 100 away loses %.2f\n",
            h, m, 10m, 100m)
end
B = alibi_bias(4, 5)
println("\n  the bias matrix for head 1 (rows = query, columns = key, -Inf is the future):\n")
for i in 1:5
    println("    ", join((isfinite(B[i, j, 1]) ? @sprintf("%6.2f", B[i, j, 1]) : "  -Inf")
                         for j in 1:5), "")
end

section("how they compare")
S = position_similarity(sinusoidal_encoding(64, 64))
d = rope_similarity(64, 64)
@printf("  gap      sinusoidal cosine    rotary dot product\n")
for g in (0, 1, 2, 4, 8, 16, 32, 63)
    @printf("  %3d          %+.3f                %+.3f\n", g, S[1, g + 1], d[g + 1])
end

save(joinpath(figs, "10_positional_encoding.png"),
     plot_positional_encoding(sinusoidal_encoding(64, 100)))
save(joinpath(figs, "11_position_decay.png"), plot_position_decay(; dim = 64, len = 64))
println("\nwrote 10_positional_encoding.png and 11_position_decay.png to ", figs)
