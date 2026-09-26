# Plotting lives in an extension so the core package stays dependency-free.
# Loading any Makie backend (CairoMakie, GLMakie, ...) activates the real methods.

const _PLOT_HINT = """
Plotting needs a Makie backend. Run

    using CairoMakie      # or GLMakie for an interactive window

before calling this function."""

for f in (:plot_tables, :plot_update, :plot_loss, :plot_similarity_matrix,
          :plot_cooccurrence, :plot_embedding_map, :plot_evolution,
          :plot_positional_encoding, :plot_position_decay,
          :plot_frequency_ladder, :plot_gap_kernel, :plot_rope_geometry,
          :plot_alibi_kernel, :plot_bipartite_attention, :plot_binary_analogy)
    @eval begin
        function $f(args...; kwargs...)
            error($_PLOT_HINT)
        end
    end
end

@doc """
    plot_tables(model; values=true, highlight=nothing, colorrange=nothing, colorbar=false)

Draw `W` above `C` as viridis heatmaps, one column per word, with the rounded
value in each cell when `values` is true. `highlight` may be an
[`UpdateInfo`](@ref), in which case the columns that update touched are outlined.

The colour scale is not drawn by default: with every cell labelled it adds
little and costs width. Pass `colorbar = true` for it.

Requires a Makie backend.
""" plot_tables

@doc """
    plot_update(model, info)

The anatomy of one update: the four questions with their guesses and errors, and
the rank-one change each `C` column receives.

Requires a Makie backend.
""" plot_update

@doc """
    plot_loss(log; targets=nothing)

Average loss per pair against epoch, with the coin-flip level `4ln2` marked.

Requires a Makie backend.
""" plot_loss

@doc """
    plot_similarity_matrix(model; order=nothing, labels=true, colorbar=false)

Cosine similarity between every pair of word vectors as a heatmap. Pass `order`
to group words and reveal block structure, and `colorbar = true` for the scale
(values run from -1 to +1 throughout).

Requires a Makie backend.
""" plot_similarity_matrix

@doc """
    plot_cooccurrence(N, vocab; order=nothing)

The co-occurrence table as a dot matrix, area proportional to the count: the
complete list of what the model will ever be told.

Requires a Makie backend.
""" plot_cooccurrence

@doc """
    plot_embedding_map(model; basis=nothing, labels=true, groups=nothing)

The word vectors flattened to two dimensions with [`pca2`](@ref) and labelled.

Requires a Makie backend.
""" plot_embedding_map

@doc """
    plot_evolution(model, snapshots; epochs=sort(collect(keys(snapshots))), groups=nothing)

A row of small maps, one per snapshot, all projected onto the axes of the last
one so the motion between frames is comparable.

Requires a Makie backend.
""" plot_evolution

@doc """
    plot_positional_encoding(P; colorbar=false)

The encoding itself as a heatmap: dimensions down, positions across. The fast
clocks are the top rows, the slow ones the bottom, and reading a column is
reading the position.

Requires a Makie backend.
""" plot_positional_encoding

@doc """
    plot_position_decay(; dim=64, len=64, nheads=4)

How each scheme scores distance: the cosine between sinusoidal position vectors,
the rotary dot product from [`rope_similarity`](@ref), and the ALiBi penalty per
head, all against the gap between two positions.

Requires a Makie backend.
""" plot_position_decay

@doc """
    plot_frequency_ladder(; dim=64, base=10000.0, context=2048)

The geometric ladder of wavelengths behind the sinusoidal and rotary schemes:
one bar per coordinate pair, from a few positions to tens of thousands. Pairs
whose wavelength is shorter than `context` have wrapped at least once within a
window that long, and can no longer place a token on their own.

Requires a Makie backend.
""" plot_frequency_ladder

@doc """
    plot_gap_kernel(; dim=64, base=10000.0, len=2000)

The exact sinusoidal similarity kernel `S(g) = Σᵢ cos(g θᵢ)` against the smooth
logarithmic envelope, on a log axis. The point of the figure is the gap between
the two: the kernel is a clean decay only for the first few positions, and
oscillates well away from the envelope afterwards.

Requires a Makie backend.
""" plot_gap_kernel

@doc """
    plot_rope_geometry(; dim=64, base=10000.0, seed=1)

Three views of the rotary scheme: one coordinate pair rotated to two different
positions in the plane, the per-pair decomposition of a score into amplitude
and phase from [`rope_channels`](@ref), and the resulting score against the gap.

Requires a Makie backend.
""" plot_rope_geometry

@doc """
    plot_alibi_kernel(; nheads=8, len=256)

ALiBi on the scale the softmax sees: the multiplicative discount `exp(-mₕ d)`
per head, with each head's half-life marked. The half-lives form a geometric
ladder, which is how a handful of heads cover distances from a couple of tokens
to a couple of hundred.

Requires a Makie backend.
""" plot_alibi_kernel

@doc """
    plot_bipartite_attention(; heads=(1, 6), nheads=8, len=10)

Attention drawn as what it is: a weighted bipartite graph, queries along the
top and keys along the bottom, one edge per pair. Edge opacity is the ALiBi
discount `exp(-mₕ|i-j|)` for the chosen heads, so a steep head shows as a tight
band along the diagonal and a shallow one as a broad fan. Only the causal half
is drawn, since a token cannot attend to its future.

Requires a Makie backend.
""" plot_bipartite_attention

@doc """
    plot_binary_analogy(; dim=8, len=64, base=10000.0, ladder_dim=64)

Six panels comparing a binary counter with a sinusoidal encoding.

Top row: the codes themselves — plain binary, reflected Gray, and sinusoidal —
all showing the same geometric ladder of frequencies, square waves against
sines. Bottom row: what each says about *distance*. The binary Hamming matrix
is a self-similar block pattern that is not constant along its diagonals; the
sinusoidal similarity matrix is a clean band, because its score depends on the
gap alone. The last panel puts the two wavelength ladders side by side at a realistic
width (`ladder_dim`, 32 channels each), rather than at the small `dim` used to
keep the code panels legible.

Requires a Makie backend.
""" plot_binary_analogy
