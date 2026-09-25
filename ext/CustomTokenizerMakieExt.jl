module CustomTokenizerMakieExt

using CustomTokenizer
using CustomTokenizer: Model, UpdateInfo, TrainLog, Vocabulary, pca2,
                           similarity_matrix, column_norms, changed_columns
using Makie
using LinearAlgebra, Printf

const ANIMAL = RGBf(0.106, 0.620, 0.467)
const ROYAL  = RGBf(0.459, 0.439, 0.702)
const NEUTRAL = RGBf(0.549, 0.549, 0.549)
const CENTER_EDGE = RGBf(0.95, 0.10, 0.35)
const CTX_EDGE = RGBf(1.0, 0.55, 0.0)

celllabel(v) = (a = abs(v);
                s = a < 1 ? replace(@sprintf("%.3f", a), "0." => ".") : @sprintf("%.2f", a);
                (v < -5e-4 ? "−" : "") * s)

ink_on(t) = t < 0.62 ? RGBf(1, 1, 1) : RGBf(0.05, 0.05, 0.05)

"Draw one labelled heatmap of a dim × V table into `ax`."
function _table!(ax, M, words, crange; values = true, outline = Int[], edge = CTX_EDGE)
    d, V = size(M)
    heatmap!(ax, 1:V, 1:d, permutedims(M); colormap = :viridis, colorrange = crange)
    if values
        lo, hi = crange
        pos = [Point2f(j, r) for j in 1:V for r in 1:d]
        txt = [celllabel(M[r, j]) for j in 1:V for r in 1:d]
        col = [ink_on((M[r, j] - lo) / (hi - lo)) for j in 1:V for r in 1:d]
        text!(ax, pos; text = txt, color = col, align = (:center, :center),
              fontsize = V > 30 ? 5 : 9)
    end
    for j in outline
        poly!(ax, Rect2f(j - 0.5, 0.5, 1, d); color = :transparent, strokecolor = edge,
              strokewidth = 3)
    end
    ax.xticks = (1:V, words)
    ax.yticks = (1:d, string.(1:d))
    ax.xticklabelrotation = pi / 4
    ax.xticklabelsize = V > 30 ? 7 : 11
    ax.yticklabelsize = 9
    ax.yreversed = true
    ax.xaxisposition = :top
    hidespines!(ax)
    hidedecorations!(ax; ticklabels = false)
    ax
end

function CustomTokenizer.plot_tables(m::Model; values = true, highlight = nothing,
                                         colorrange = nothing, figure = (;))
    d, V = size(m.W)
    hi = colorrange === nothing ?
         max(0.0625, maximum(abs, m.W), maximum(abs, m.C)) : colorrange
    crange = hi isa Tuple ? hi : (-hi, hi)
    wcols = highlight === nothing ? Int[] : changed_columns(highlight).W
    ccols = highlight === nothing ? Int[] : changed_columns(highlight).C
    fig = Figure(; size = (max(900, 55V), 260 + 42d), figure...)
    axW = Axis(fig[1, 1]; title = "W — word vectors (kept)", titlealign = :left,
               aspect = DataAspect())
    _table!(axW, m.W, m.vocab.words, crange; values, outline = wcols, edge = CENTER_EDGE)
    axC = Axis(fig[2, 1]; title = "C — context vectors (discarded after training)",
               titlealign = :left, aspect = DataAspect())
    _table!(axC, m.C, m.vocab.words, crange; values, outline = ccols, edge = CTX_EDGE)
    Colorbar(fig[1:2, 2]; colormap = :viridis, colorrange = crange, width = 14,
             label = "value")
    fig
end

function CustomTokenizer.plot_update(m::Model, u::UpdateInfo; figure = (;))
    d = length(u.ΔW)
    fig = Figure(; size = (980, 150 + 60d), figure...)
    Label(fig[1, 1:2], "center $(u.words[1]) · context $(u.words[2]) · " *
          "negatives " * join(u.words[3:end], ", ") * @sprintf("  ·  lr %.4f", u.lr);
          fontsize = 16, font = :bold, halign = :left)

    ax1 = Axis(fig[2, 1]; title = "the four questions", titlealign = :left,
               xlabel = "probability", yreversed = true)
    ys = 1:length(u.wants)
    barplot!(ax1, ys, u.probabilities; direction = :x,
             color = [w == 1 ? ANIMAL : CTX_EDGE for w in u.wants])
    scatter!(ax1, u.wants, ys; color = :black, marker = :vline, markersize = 18)
    ax1.yticks = (ys, [@sprintf("%s (g %+.4f)", w, g) for (w, g) in zip(u.words[2:end], u.errors)])
    xlims!(ax1, 0, 1.05)

    ax2 = Axis(fig[2, 2]; title = "ΔC[:, j] = g × W[:, center]  (one pattern, four sizes)",
               titlealign = :left, xlabel = "row of the column", ylabel = "change")
    w = CustomTokenizer.wordvec(m, u.center)
    for (k, g) in enumerate(u.errors)
        lines!(ax2, 1:d, g .* w; color = u.wants[k] == 1 ? ANIMAL : CTX_EDGE,
               linewidth = u.wants[k] == 1 ? 3 : 1.5)
    end
    lines!(ax2, 1:d, u.ΔW; color = CENTER_EDGE, linewidth = 3, linestyle = :dash)
    axislegend(ax2, [LineElement(color = ANIMAL, linewidth = 3),
                     LineElement(color = CTX_EDGE, linewidth = 1.5),
                     LineElement(color = CENTER_EDGE, linewidth = 3, linestyle = :dash)],
               ["real neighbour", "random words", "ΔW of the center"]; position = :rb,
               framevisible = false, labelsize = 10)
    fig
end

function CustomTokenizer.plot_loss(trainlog::TrainLog; figure = (;))
    losses = trainlog.losses
    coinflip = 4 * log(2)
    fig = Figure(; size = (760, 420), figure...)
    ax = Axis(fig[1, 1]; xlabel = "epoch", ylabel = "average loss per pair",
              title = "training loss", titlealign = :left)
    lines!(ax, 1:length(losses), losses; color = RGBf(0.18, 0.43, 0.71), linewidth = 2)
    hlines!(ax, [coinflip]; color = (:black, 0.35), linestyle = :dash)
    text!(ax, length(losses) * 0.55, coinflip + 0.04;
          text = "coin-flip guessing: 4 ln 2 = 2.77", fontsize = 11, color = (:black, 0.55))
    fig
end

function CustomTokenizer.plot_similarity_matrix(m::Model; order = nothing,
                                                    labels = true, figure = (;))
    idx = order === nothing ? collect(1:length(m.vocab)) : collect(order)
    S = similarity_matrix(m; order = idx)
    words = m.vocab.words[idx]
    V = length(idx)
    fig = Figure(; size = (620, 620), figure...)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true,
              title = "cosine similarity between word vectors", titlealign = :left)
    heatmap!(ax, 1:V, 1:V, permutedims(S); colormap = :viridis, colorrange = (-1, 1))
    if labels
        ax.xticks = (1:V, words); ax.yticks = (1:V, words)
        ax.xticklabelrotation = pi / 3
        ax.xticklabelsize = V > 30 ? 6 : 10
        ax.yticklabelsize = V > 30 ? 6 : 10
        ax.xaxisposition = :top
    else
        hidedecorations!(ax)
    end
    hidespines!(ax)
    Colorbar(fig[1, 2]; colormap = :viridis, colorrange = (-1, 1), width = 14)
    fig
end

function CustomTokenizer.plot_cooccurrence(N::AbstractMatrix{<:Integer}, vocab::Vocabulary;
                                               order = nothing, figure = (;))
    idx = order === nothing ? collect(1:length(vocab)) : collect(order)
    V = length(idx)
    words = vocab.words[idx]
    fig = Figure(; size = (680, 680), figure...)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true,
              title = "how often each word is the neighbour of each other word",
              titlealign = :left)
    xs = Float64[]; ys = Float64[]; ms = Float64[]; cs = Int[]
    for (a, i) in enumerate(idx), (b, j) in enumerate(idx)
        n = N[i, j]
        n == 0 && continue
        push!(xs, b); push!(ys, a); push!(ms, 6 + 4n); push!(cs, n)
    end
    scatter!(ax, xs, ys; markersize = ms, color = cs, colormap = :viridis)
    ax.xticks = (1:V, words); ax.yticks = (1:V, words)
    ax.xticklabelrotation = pi / 3
    ax.xticklabelsize = V > 30 ? 6 : 10
    ax.yticklabelsize = V > 30 ? 6 : 10
    ax.xaxisposition = :top
    hidespines!(ax)
    fig
end

_groupcolor(g) = g === :animal ? ANIMAL : g === :royal ? ROYAL : NEUTRAL

function CustomTokenizer.plot_embedding_map(m::Model; basis = nothing, centre = nothing,
                                                labels = true, groups = nothing, figure = (;))
    P, B, c = pca2(m.W; basis, centre)
    fig = Figure(; size = (620, 600), figure...)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), title = "word vectors in two dimensions",
              titlealign = :left)
    cols = groups === nothing ? fill(NEUTRAL, length(m.vocab)) :
           [_groupcolor(get(groups, w, :other)) for w in m.vocab.words]
    scatter!(ax, P[1, :], P[2, :]; color = cols, markersize = 10)
    if labels
        text!(ax, P[1, :], P[2, :]; text = m.vocab.words, fontsize = 10,
              align = (:left, :center), offset = (6, 0), color = cols)
    end
    hidedecorations!(ax); hidespines!(ax)
    fig
end

function CustomTokenizer.plot_evolution(m::Model, snapshots::AbstractDict;
                                            epochs = sort(collect(keys(snapshots))),
                                            groups = nothing, figure = (;))
    final = snapshots[maximum(epochs)].W
    _, B, c = pca2(final)
    n = length(epochs)
    fig = Figure(; size = (250n, 300), figure...)
    cols = groups === nothing ? fill(NEUTRAL, length(m.vocab)) :
           [_groupcolor(get(groups, w, :other)) for w in m.vocab.words]
    for (k, e) in enumerate(epochs)
        P, _, _ = pca2(snapshots[e].W; basis = B, centre = c)
        ax = Axis(fig[1, k]; aspect = DataAspect(),
                  title = e == 0 ? "before training" : "epoch $e", titlesize = 13)
        scatter!(ax, P[1, :], P[2, :]; color = cols, markersize = 7)
        limits!(ax, -0.95, 0.95, -0.95, 0.95)
        hidedecorations!(ax); hidespines!(ax)
    end
    fig
end

end # module
