module CustomTokenizerMakieExt

using CustomTokenizer
using CustomTokenizer: Model, UpdateInfo, TrainLog, Vocabulary, pca2,
                           similarity_matrix, column_norms, changed_columns,
                           sinusoidal_encoding, position_similarity, rope_similarity,
                           alibi_slopes, sinusoidal_frequencies, sinusoidal_wavelengths,
                           sinusoidal_gap_score, sinusoidal_gap_envelope, rope_channels,
                           alibi_halflife, alibi_decay, rope, binary_encoding,
                           binary_wavelengths, hamming_matrix, pairs_per_octave
using Makie
using LinearAlgebra, Printf, Random, Statistics

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
                                         colorrange = nothing, colorbar = false,
                                         figure = (;))
    d, V = size(m.W)
    hi = colorrange === nothing ?
         max(0.0625, maximum(abs, m.W), maximum(abs, m.C)) : colorrange
    crange = hi isa Tuple ? hi : (-hi, hi)
    wcols = highlight === nothing ? Int[] : changed_columns(highlight).W
    ccols = highlight === nothing ? Int[] : changed_columns(highlight).C
    fig = Figure(; size = (max(860, 52V) + (colorbar ? 70 : 0), 250 + 42d), figure...)
    axW = Axis(fig[1, 1]; title = "W — word vectors (kept)", titlealign = :left,
               aspect = DataAspect())
    _table!(axW, m.W, m.vocab.words, crange; values, outline = wcols, edge = CENTER_EDGE)
    axC = Axis(fig[2, 1]; title = "C — context vectors (discarded after training)",
               titlealign = :left, aspect = DataAspect())
    _table!(axC, m.C, m.vocab.words, crange; values, outline = ccols, edge = CTX_EDGE)
    colorbar && Colorbar(fig[1:2, 2]; colormap = :viridis, colorrange = crange,
                         width = 14, label = "value")
    rowgap!(fig.layout, 10)
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
                                                    labels = true, colorbar = false,
                                                    figure = (;))
    idx = order === nothing ? collect(1:length(m.vocab)) : collect(order)
    S = similarity_matrix(m; order = idx)
    words = m.vocab.words[idx]
    V = length(idx)
    fig = Figure(; size = (620 + (colorbar ? 60 : 0), 620), figure...)
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
    colorbar && Colorbar(fig[1, 2]; colormap = :viridis, colorrange = (-1, 1), width = 14)
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

function CustomTokenizer.plot_positional_encoding(P::AbstractMatrix; colorbar = false,
                                                  figure = (;))
    d, n = size(P)
    fig = Figure(; size = (760 + (colorbar ? 70 : 0), 380), figure...)
    ax = Axis(fig[1, 1]; xlabel = "position", ylabel = "dimension",
              title = "sinusoidal encoding: fast clocks on top, slow ones below",
              titlealign = :left, yreversed = true)
    heatmap!(ax, 1:n, 1:d, permutedims(P); colormap = :viridis, colorrange = (-1, 1))
    colorbar && Colorbar(fig[1, 2]; colormap = :viridis, colorrange = (-1, 1), width = 14)
    fig
end

function CustomTokenizer.plot_position_decay(; dim = 64, len = 64, nheads = 4, figure = (;))
    fig = Figure(; size = (1080, 380), figure...)
    gaps = 0:(len - 1)

    P = sinusoidal_encoding(dim, len)
    S = position_similarity(P)
    ax1 = Axis(fig[1, 1]; xlabel = "gap between positions", ylabel = "cosine",
               title = "sinusoidal: position vectors", titlealign = :left)
    lines!(ax1, gaps, [S[1, g + 1] for g in gaps]; color = ANIMAL, linewidth = 2,
           label = "from position 0")
    lines!(ax1, 0:(len - 21), [S[21, 21 + g] for g in 0:(len - 21)]; color = ROYAL,
           linewidth = 2, linestyle = :dash, label = "from position 20")
    axislegend(ax1; position = :rt, framevisible = false, labelsize = 10)

    ax2 = Axis(fig[1, 2]; xlabel = "gap between positions", ylabel = "dot product",
               title = "rotary: a token against itself", titlealign = :left)
    lines!(ax2, gaps, rope_similarity(dim, len); color = CTX_EDGE, linewidth = 2)

    ax3 = Axis(fig[1, 3]; xlabel = "gap between positions", ylabel = "added to the score",
               title = "ALiBi: one penalty per head", titlealign = :left)
    for (h, m) in enumerate(alibi_slopes(nheads))
        lines!(ax3, gaps, [-m * g for g in gaps]; linewidth = 2,
               label = @sprintf("slope %.3f", m))
    end
    axislegend(ax3; position = :lb, framevisible = false, labelsize = 10)
    fig
end

function CustomTokenizer.plot_frequency_ladder(; dim = 64, base = 10_000.0,
                                               context = 2048, figure = (;))
    λ = sinusoidal_wavelengths(dim; base = base)
    npairs = length(λ)
    fig = Figure(; size = (1080, 400), figure...)

    ax1 = Axis(fig[1, 1]; xlabel = "coordinate pair i", ylabel = "wavelength (positions)",
               yscale = log10, title = "each pair turns at its own rate",
               titlealign = :left)
    wrapped = λ .< context
    barplot!(ax1, 1:npairs, λ; color = [w ? CTX_EDGE : ANIMAL for w in wrapped])
    hlines!(ax1, [context]; color = :black, linestyle = :dash, linewidth = 1.5)
    text!(ax1, 1, context; text = "  context = $context", align = (:left, :bottom),
          fontsize = 11)
    text!(ax1, npairs, minimum(λ); text = "orange: wraps inside the context  ",
          align = (:right, :bottom), fontsize = 10, color = CTX_EDGE)

    # how many pairs can still be read unambiguously at a given gap
    gaps = [1, 10, 100, 1000, 10_000]
    θ = sinusoidal_frequencies(dim; base = base)
    unwrapped = [count(t -> g * t < π, θ) for g in gaps]
    ax2 = Axis(fig[1, 2]; xlabel = "gap between two tokens", ylabel = "pairs not yet wrapped",
               xscale = log10, xticks = (gaps, string.(gaps)),
               title = "how many clocks still read unambiguously", titlealign = :left)
    barplot!(ax2, gaps, unwrapped; color = ROYAL, width = [0.6g for g in gaps])
    for (g, u) in zip(gaps, unwrapped)
        text!(ax2, g, u; text = "$u/$npairs", align = (:center, :bottom), fontsize = 11)
    end
    ylims!(ax2, 0, npairs * 1.15)
    fig
end

function CustomTokenizer.plot_gap_kernel(; dim = 64, base = 10_000.0, len = 2000,
                                         figure = (;))
    gaps = 1:len
    S = sinusoidal_gap_score(dim, gaps; base = base)
    E = sinusoidal_gap_envelope(dim, gaps; base = base)
    monotone = something(findfirst(g -> S[g + 1] > S[g], 1:(len - 1)), len)

    fig = Figure(; size = (1080, 400), figure...)
    ax1 = Axis(fig[1, 1]; xlabel = "gap", ylabel = "S(gap)",
               title = "the first few positions: a clean decay", titlealign = :left)
    small = 1:min(40, len)
    lines!(ax1, small, S[small]; color = ANIMAL, linewidth = 2.5)
    scatter!(ax1, small, S[small]; color = ANIMAL, markersize = 5)
    vlines!(ax1, [monotone]; color = :black, linestyle = :dash, linewidth = 1.5)
    text!(ax1, monotone, maximum(S[small]);
          text = " monotone only to gap $monotone", align = (:left, :top), fontsize = 11)

    ax2 = Axis(fig[1, 2]; xlabel = "gap (log scale)", ylabel = "S(gap)", xscale = log10,
               title = "further out: oscillation, not decay", titlealign = :left)
    lines!(ax2, gaps, S; color = ANIMAL, linewidth = 1.5, label = "exact  Σᵢ cos(g θᵢ)")
    lines!(ax2, gaps, E; color = ROYAL, linewidth = 2.5, linestyle = :dash,
           label = "log envelope")
    hlines!(ax2, [0]; color = :black, linewidth = 0.8)
    axislegend(ax2; position = :lb, framevisible = false, labelsize = 10)
    fig
end

function CustomTokenizer.plot_rope_geometry(; dim = 64, base = 10_000.0, seed = 1,
                                            figure = (;))
    rng = Xoshiro(seed)
    q, k = randn(rng, dim), randn(rng, dim)
    fig = Figure(; size = (1120, 390), figure...)

    # --- one pair, rotated to two positions: the angle between them is what survives
    ax1 = Axis(fig[1, 1]; title = "one coordinate pair, in its plane", titlealign = :left,
               aspect = DataAspect(), xlabel = "x₀", ylabel = "x₁")
    θ0 = sinusoidal_frequencies(dim; base = base)[1]
    z = [q[1], q[2]] ./ norm(q[1:2])
    w = [k[1], k[2]] ./ norm(k[1:2])
    rot(v, a) = [cos(a) * v[1] - sin(a) * v[2], sin(a) * v[1] + cos(a) * v[2]]
    arc = [Point2f(cos(t), sin(t)) for t in range(0, 2π; length = 200)]
    lines!(ax1, arc; color = (:black, 0.15))
    for (m, col) in ((0, ANIMAL), (6, ROYAL))
        zm, wm = rot(z, m * θ0), rot(w, m * θ0)
        arrows2d!(ax1, [0.0, 0.0], [0.0, 0.0], [zm[1], wm[1]], [zm[2], wm[2]];
                  color = col)
        text!(ax1, zm[1], zm[2]; text = m == 0 ? " q at 0" : " q at 6", color = col,
              fontsize = 11, align = (:left, :bottom))
        text!(ax1, wm[1], wm[2]; text = m == 0 ? " k at 0" : " k at 6", color = col,
              fontsize = 11, align = (:left, :bottom))
    end
    limits!(ax1, -1.35, 1.35, -1.35, 1.35)
    text!(ax1, 0, -1.3; text = "both turn together: the angle between them never changes",
          fontsize = 10, align = (:center, :bottom), color = NEUTRAL)

    # --- the score as a bank of cosines, one per pair
    ch = rope_channels(q, k; base = base)
    ax2 = Axis(fig[1, 2]; xlabel = "coordinate pair i", ylabel = "amplitude |zᵢ||wᵢ|",
               title = "content sets amplitude and phase", titlealign = :left)
    barplot!(ax2, 1:length(ch.amplitude), ch.amplitude; color = ANIMAL)
    ax2b = Axis(fig[1, 2]; yaxisposition = :right, ylabel = "phase φᵢ (rad)",
                yticklabelcolor = CTX_EDGE, ylabelcolor = CTX_EDGE)
    hidespines!(ax2b); hidexdecorations!(ax2b)
    scatter!(ax2b, 1:length(ch.phase), ch.phase; color = CTX_EDGE, markersize = 6)
    linkxaxes!(ax2, ax2b)

    # --- the resulting score against the gap
    ax3 = Axis(fig[1, 3]; xlabel = "gap n − m", ylabel = "score (normalised)",
               title = "decay is an average, not a promise", titlealign = :left)
    gaps = 0:255
    one = [dot(rope(q, 0; base = base), rope(k, g; base = base)) for g in gaps] /
          (norm(q) * norm(k))
    lines!(ax3, gaps, one; color = (ROYAL, 0.55), linewidth = 1.2,
           label = "one random q, k")
    vs = [randn(Xoshiro(1000 + j), dim) for j in 1:128]
    same = [mean(dot(rope(v, 0; base = base), rope(v, g; base = base)) / dot(v, v)
                 for v in vs) for g in gaps]
    lines!(ax3, gaps, same; color = ANIMAL, linewidth = 2.5,
           label = "a token against itself, averaged")
    hlines!(ax3, [0]; color = :black, linewidth = 0.8)
    axislegend(ax3; position = :rt, framevisible = false, labelsize = 9)
    fig
end

function CustomTokenizer.plot_alibi_kernel(; nheads = 8, len = 256, figure = (;))
    D = alibi_decay(nheads, len)
    half = alibi_halflife(nheads)
    slopes = alibi_slopes(nheads)
    fig = Figure(; size = (1080, 400), figure...)

    ax1 = Axis(fig[1, 1]; xlabel = "distance |i − j|", ylabel = "attention multiplier",
               title = "after the softmax, the bias is a geometric discount",
               titlealign = :left)
    headcolor(h) = Makie.get(Makie.cgrad(:viridis), nheads == 1 ? 0.5 : (h - 1) / (nheads - 1))
    for h in 1:nheads
        lines!(ax1, 0:(len - 1), D[:, h]; linewidth = 2, color = headcolor(h),
               label = @sprintf("m = %.4f  (half-life %.0f)", slopes[h], half[h]))
    end
    hlines!(ax1, [0.5]; color = :black, linestyle = :dash, linewidth = 1)
    text!(ax1, len, 0.5; text = "half  ", align = (:right, :bottom), fontsize = 10)
    xlims!(ax1, 0, min(len, 60))
    axislegend(ax1; position = :rt, framevisible = false, labelsize = 9)

    ax2 = Axis(fig[1, 2]; xlabel = "head", ylabel = "half-life (positions)", yscale = log10,
               xticks = 1:nheads, title = "a geometric ladder of ranges", titlealign = :left)
    barplot!(ax2, 1:nheads, half; color = [headcolor(h) for h in 1:nheads])
    for h in 1:nheads
        text!(ax2, h, half[h];
              text = half[h] < 10 ? @sprintf("%.1f", half[h]) : @sprintf("%.0f", half[h]),
              align = (:center, :bottom), fontsize = 10)
    end
    fig
end

function CustomTokenizer.plot_bipartite_attention(; heads = (1, 6), nheads = 8, len = 10,
                                                  figure = (;))
    slopes = alibi_slopes(nheads)
    half = alibi_halflife(nheads)
    fig = Figure(; size = (1080, 400), figure...)
    for (panel, h) in enumerate(heads)
        m = slopes[h]
        ax = Axis(fig[1, panel];
                  title = @sprintf("head %d:  m = %.4f,  half-life %.1f positions",
                                   h, m, half[h]),
                  titlealign = :left)
        hidedecorations!(ax); hidespines!(ax)
        for i in 1:len, j in 1:i                      # causal: key j ≤ query i
            w = exp(-m * (i - j))
            lines!(ax, [i, j], [1.0, 0.0];
                   color = (ROYAL, max(w, 0.02)), linewidth = 0.5 + 3.5w)
        end
        scatter!(ax, 1:len, fill(1.0, len); color = ANIMAL, markersize = 13)
        scatter!(ax, 1:len, fill(0.0, len); color = CTX_EDGE, markersize = 13)
        text!(ax, 0.4, 1.0; text = "queries ", align = (:right, :center), fontsize = 11)
        text!(ax, 0.4, 0.0; text = "keys ", align = (:right, :center), fontsize = 11)
        for i in 1:len
            text!(ax, i, 1.12; text = string(i), align = (:center, :center), fontsize = 9)
            text!(ax, i, -0.12; text = string(i), align = (:center, :center), fontsize = 9)
        end
        limits!(ax, -1.6, len + 0.6, -0.35, 1.35)
    end
    fig
end

function CustomTokenizer.plot_binary_analogy(; dim = 8, len = 64, base = 10_000.0,
                                             ladder_dim = 64, figure = (;))
    B = binary_encoding(dim, len)
    G = binary_encoding(dim, len; gray = true)
    P = sinusoidal_encoding(dim, len; base = base)
    fig = Figure(; size = (1150, 640), figure...)

    function codepanel(pos, M, title; two_tone = true)
        ax = Axis(fig[pos...]; title = title, titlealign = :left, yreversed = true,
                  xlabel = "position", ylabel = "channel")
        heatmap!(ax, 1:size(M, 2), 1:size(M, 1), permutedims(M);
                 colormap = two_tone ? :binary : :viridis,
                 colorrange = two_tone ? (0, 1) : (-1, 1))
        ax
    end
    codepanel((1, 1), B, "binary: bit k is a square wave of period 2ᵏ⁺¹")
    codepanel((1, 2), G, "Gray: neighbours always differ in one bit")
    codepanel((1, 3), P, "sinusoidal: the same ladder, smoothed"; two_tone = false)

    H = hamming_matrix(B)
    ax4 = Axis(fig[2, 1]; title = "binary distance: not constant along diagonals",
               titlealign = :left, xlabel = "position", ylabel = "position",
               yreversed = true)
    heatmap!(ax4, 1:len, 1:len, permutedims(H); colormap = :magma)

    S = position_similarity(P)
    ax5 = Axis(fig[2, 2]; title = "sinusoidal similarity: a band (Toeplitz)",
               titlealign = :left, xlabel = "position", ylabel = "position",
               yreversed = true)
    heatmap!(ax5, 1:len, 1:len, permutedims(S); colormap = :viridis)

    # the ladder panel uses a production width, not the small display width:
    # 32 channels each, which is what the real comparison looks like
    nch = ladder_dim ÷ 2
    ax6 = Axis(fig[2, 3]; yscale = log10, xlabel = "channel", ylabel = "wavelength",
               title = @sprintf("%d channels each: %.2f pairs per octave vs 1",
                                nch, pairs_per_octave(ladder_dim; base = base)),
               titlealign = :left)
    scatterlines!(ax6, 1:nch, binary_wavelengths(nch); color = CTX_EDGE, markersize = 6,
                  label = @sprintf("binary bits (reach 2^%d)", nch))
    λ = 2π ./ [base^(-2i / ladder_dim) for i in 0:(nch - 1)]
    scatterlines!(ax6, 1:nch, λ; color = ANIMAL, markersize = 6,
                  label = @sprintf("sinusoidal pairs (reach %.0f)", maximum(λ)))
    axislegend(ax6; position = :lt, framevisible = false, labelsize = 9)
    fig
end

end # module
