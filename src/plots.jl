# Plotting lives in an extension so the core package stays dependency-free.
# Loading any Makie backend (CairoMakie, GLMakie, ...) activates the real methods.

const _PLOT_HINT = """
Plotting needs a Makie backend. Run

    using CairoMakie      # or GLMakie for an interactive window

before calling this function."""

for f in (:plot_tables, :plot_update, :plot_loss, :plot_similarity_matrix,
          :plot_cooccurrence, :plot_embedding_map, :plot_evolution)
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
