# Generate every figure the package can draw into examples/figures/.
#
# Run:  julia --project=.. figures.jl

using CustomTokenizer, CairoMakie, Printf

CairoMakie.activate!(type = "png")
outdir = joinpath(@__DIR__, "figures")
mkpath(outdir)
save_fig(name, fig) = (path = joinpath(outdir, name); save(path, fig); println("wrote ", path))

sentences = tokenize(TOY_TEXT)
vocab = build_vocab(sentences)
model = Model(vocab)
pairs = corpus_pairs(sentences, vocab; window = 2)
N = cooccurrence(sentences, vocab; window = 2)

groups = Dict(w => (w in ("the", "a") ? :other :
                    w in ("king", "queen", "kingdom", "rules", "wears", "crown",
                          "lives", "in", "castle") ? :royal : :animal)
              for w in vocab.words)
order = sortperm(1:length(vocab);
                 by = i -> (groups[vocab[i]] === :animal ? 1 :
                            groups[vocab[i]] === :royal ? 2 : 3, i))

save_fig("01_tables_start.png", plot_tables(model))
save_fig("02_cooccurrence.png", plot_cooccurrence(N, vocab; order = order))

info = update_pair!(model, "cat", "drinks", ["on", "wears", "queen"], 0.5)
save_fig("03_tables_after_one_update.png", plot_tables(model; highlight = info))
save_fig("04_update_anatomy.png", plot_update(model, info))

trainlog = train!(model, pairs; epochs = 300, snapshots = [0, 1, 5, 10, 20, 50, 100, 300])
save_fig("05_loss.png", plot_loss(trainlog))
save_fig("06_tables_trained.png", plot_tables(model; values = false))
save_fig("07_similarity.png", plot_similarity_matrix(model; order = order))
save_fig("08_map.png", plot_embedding_map(model; groups = groups))
save_fig("09_evolution.png", plot_evolution(model, trainlog.snapshots; groups = groups))

@printf("\ndone: %d figures in %s\n", length(readdir(outdir)), outdir)
