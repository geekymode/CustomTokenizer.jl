# Plot the output of tokenizer_zoo.py.
#
#   python experiments/tokenizer_zoo.py      # writes experiments/results/*.tsv
#   julia --project=docs experiments/tokenizer_plots.jl

using CairoMakie, DelimitedFiles, Printf

CairoMakie.activate!(type = "png")
here = @__DIR__
results = joinpath(here, "results")
figs = joinpath(here, "figures"); mkpath(figs)

raw, header = readdlm(joinpath(results, "fertility.tsv"), '\t'; header = true)
names_ = vec(string.(header))
texts = string.(raw[:, 1])
bytes_ = Float64.(raw[:, 2])
toknames = names_[4:end]
counts = Float64.(raw[:, 4:end])                 # texts × tokenizers

vraw, _ = readdlm(joinpath(results, "vocab.tsv"), '\t'; header = true)
vnames, vsizes = string.(vraw[:, 1]), Float64.(vraw[:, 2])

palette = Makie.wong_colors()
colour(i) = palette[mod1(i, length(palette))]

# ---------------------------------------------------------------- tokens per text
fig = Figure(size = (1150, 560))
ax = Axis(fig[1, 1]; xticks = (1:length(texts), texts), ylabel = "tokens",
          title = "How many tokens each tokenizer needs for the same text",
          titlealign = :left)
n = length(toknames)
for (k, name) in enumerate(toknames)
    barplot!(ax, (1:length(texts)) .+ (k - (n + 1) / 2) * 0.1, counts[:, k];
             width = 0.1, color = colour(k), label = name)
end
axislegend(ax; position = :lt, framevisible = false, labelsize = 11, nbanks = 2)
save(joinpath(figs, "tokens_per_text.png"), fig)

# ---------------------------------------------------------------- bytes per token
fig2 = Figure(size = (1150, 520))
ax2 = Axis(fig2[1, 1]; xticks = (1:length(texts), texts), ylabel = "bytes per token",
           title = "Bytes per token — higher means more text fits in the same context window",
           titlealign = :left)
for (k, name) in enumerate(toknames)
    name == "CustomTokenizer.jl" && continue      # not reversible, so the ratio is meaningless
    barplot!(ax2, (1:length(texts)) .+ (k - (n + 1) / 2) * 0.1, bytes_ ./ counts[:, k];
             width = 0.1, color = colour(k), label = name)
end
hlines!(ax2, [4.0]; color = (:black, 0.3), linestyle = :dash)
text!(ax2, 0.6, 4.1; text = "≈ 4 bytes/token, the usual English rule of thumb",
      fontsize = 11, color = (:black, 0.55))
axislegend(ax2; position = :rt, framevisible = false, labelsize = 11)
save(joinpath(figs, "bytes_per_token.png"), fig2)

# ---------------------------------------------------------------- vocabularies
fig3 = Figure(size = (720, 420))
ax3 = Axis(fig3[1, 1]; xticks = (1:length(vnames), vnames), ylabel = "vocabulary entries",
           title = "Vocabulary sizes", titlealign = :left, xticklabelrotation = pi / 6)
barplot!(ax3, 1:length(vnames), vsizes; color = [colour(i) for i in 1:length(vnames)])
for (i, v) in enumerate(vsizes)
    text!(ax3, i, v; text = string(Int(v)), align = (:center, :bottom), fontsize = 10)
end
save(joinpath(figs, "vocabulary_sizes.png"), fig3)

@printf("wrote 3 figures to %s\n", figs)
