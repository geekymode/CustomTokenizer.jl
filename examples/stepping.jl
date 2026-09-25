# Watch the tables fill in, one update at a time.
#
# Run:  julia --project=.. -i stepping.jl     (-i keeps the REPL open)

using CustomTokenizer, Printf, LinearAlgebra

sentences = tokenize(TOY_TEXT)
vocab = build_vocab(sentences)
model = Model(vocab)
pairs = corpus_pairs(sentences, vocab; window = 2)
stepper = TrainStepper(model, pairs; seed = 7)

"Print one line per update, plus a map of which columns are non-zero."
function trace!(s::TrainStepper, n::Integer = 1)
    for _ in 1:n
        info = step!(s)
        p = progress(s)
        filled = count(j -> !iszero(@view s.model.C[:, j]), 1:length(s.model.vocab))
        @printf("update %3d (epoch %d, pair %3d)  %-8s ← %-8s  |ΔW| %.1e  C columns filled %2d/%d\n",
                p.update, p.epoch, p.pair, info.words[1], info.words[2], norm(info.ΔW),
                filled, length(s.model.vocab))
    end
    last_update(s)
end

println("the first twelve updates:\n")
trace!(stepper, 12)

println("\nW does not move until some C column is non-zero — that is why the")
println("first |ΔW| values are exactly zero.\n")

println("running a full epoch ...")
finish_epoch!(stepper)
@printf("after epoch 1: largest |W| %.3f, largest |C| %.3f\n",
        maximum(abs, model.W), maximum(abs, model.C))

println("\nfifty more epochs ...")
for _ in 1:50
    finish_epoch!(stepper)
end
@printf("cat ~ dog %+.2f · king ~ queen %+.2f · cat ~ king %+.2f\n",
        similarity(model, "cat", "dog"), similarity(model, "king", "queen"),
        similarity(model, "cat", "king"))
println("\nthe stepper is still live: try step!(stepper), progress(stepper), reset!(stepper)")
