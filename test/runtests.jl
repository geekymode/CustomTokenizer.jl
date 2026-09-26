using CustomTokenizer
using CustomTokenizer: sigmoid, changed_columns
using Test, LinearAlgebra, Random, Statistics, JSON3

const SENTENCES = tokenize(TOY_TEXT)
const VOCAB = build_vocab(SENTENCES)
const PAIRS = corpus_pairs(SENTENCES, VOCAB; window = 2)

"A fresh untrained model on the toy corpus."
toy_model(; kwargs...) = Model(VOCAB; dim = 8, window = 2, negative = 3, kwargs...)

@testset "CustomTokenizer" begin

@testset "tokenize" begin
    s = tokenize("The cat drinks milk. The dog drinks water.")
    @test s == [["the", "cat", "drinks", "milk"], ["the", "dog", "drinks", "water"]]
    @test length(SENTENCES) == 12
    @test sum(length, SENTENCES) == 62
    # punctuation, digits and case are discarded by this tokenizer
    @test tokenize("Hello, World 42!") == [["hello", "world"]]
    # empty sentences never appear
    @test isempty(tokenize("..."))
    @test tokenize("A B"; lower = false) == [["A", "B"]]
end

@testset "vocabulary" begin
    @test length(VOCAB) == 22
    @test VOCAB[1] == "the" && VOCAB.counts[1] == 16
    @test VOCAB["cat"] == 3
    @test VOCAB[VOCAB["cat"]] == "cat"
    @test haskey(VOCAB, "queen") && !haskey(VOCAB, "aardvark")
    @test sum(VOCAB.counts) == 62
    @test vocab_size(VOCAB) == 22
    # most frequent first, ties alphabetical
    @test issorted(collect(zip(-VOCAB.counts, VOCAB.words)))
    # min_count drops the singletons
    @test length(build_vocab(SENTENCES; min_count = 2)) == 17
    @test eltype(collect(VOCAB)) == Tuple{Int,String,Int}
end

@testset "pairs and counts" begin
    @test length(context_pairs(["a", "b", "c", "d"], 2)) == 10          # 4n - 6
    @test length(context_pairs(["a", "b", "c", "d", "e"], 2)) == 14
    @test length(PAIRS) == 176
    # no pair crosses a sentence boundary: milk ends sentence 1, the starts sentence 2
    @test !((VOCAB["milk"], VOCAB["the"]) in PAIRS)
    # a word is never its own context here
    N = cooccurrence(SENTENCES, VOCAB; window = 2)
    @test sum(N) == length(PAIRS)
    @test N == N'                                                        # symmetric
    @test all(N[i, i] == 0 for i in 1:length(VOCAB))
    @test row_totals(N)[VOCAB["cat"]] == 11
    @test pair_count(N, VOCAB, "cat", "drinks") == 1
    @test pair_count(N, VOCAB, "cat", "chases") == 2
    # king and queen keep exactly the same company
    ik, iq = VOCAB["king"], VOCAB["queen"]
    @test N[ik, [1:ik-1; ik+1:iq-1; iq+1:end]] == N[iq, [1:ik-1; ik+1:iq-1; iq+1:end]]
    # cat/dog words never meet king/queen words
    @test N[VOCAB["cat"], VOCAB["crown"]] == 0
end

@testset "model construction" begin
    m = toy_model()
    @test size(m) == (8, 22)
    @test all(iszero, m.C)                      # C starts at exactly zero
    @test maximum(abs, m.W) <= 0.5 / 8          # W starts small
    # the documented column for cat, from seed 7
    @test round.(wordvec(m, "cat"); digits = 4) ==
          [0.0301, 0.0467, 0.0174, 0.0148, -0.0089, 0.0001, -0.0098, -0.0267]
    # same seed, same tables
    @test Model(VOCAB; rng = Xoshiro(7)).W == Model(VOCAB; rng = Xoshiro(7)).W
    @test Model(VOCAB; rng = Xoshiro(8)).W != Model(VOCAB; rng = Xoshiro(7)).W
    # every guess is 50/50 before training, because C is zero
    @test all(probability(m, i, j) == 0.5 for i in 1:5, j in 1:5)
    @test wordvec(m, "cat") == wordvec(m, 3)
end

@testset "negative sampling" begin
    rng = Xoshiro(1)
    V = length(VOCAB)
    negs = sample_negatives(UniformNegatives(), rng, V, (3, 10), 3)
    @test length(negs) == 3
    @test all(n -> 1 <= n <= V, negs)
    @test !(3 in negs) && !(10 in negs)
    # draws are independent, so duplicates are possible; over many draws each
    # eligible word shows up
    many = sample_negatives(UniformNegatives(), Xoshiro(2), V, (1, 2), 4000)
    @test length(unique(many)) == V - 2
    @test sampling_probabilities(UniformNegatives(), V) ≈ fill(1 / (V - 2), V)

    uni = UnigramNegatives(VOCAB)
    p = sampling_probabilities(uni, V)
    @test sum(p) ≈ 1
    @test p[VOCAB["the"]] > p[VOCAB["water"]]            # frequent words drawn more
    @test p[VOCAB["the"]] < VOCAB.counts[1] / 62         # but less than raw frequency
    draws = sample_negatives(uni, Xoshiro(3), V, (1, 2), 5000)
    @test !(1 in draws) && !(2 in draws)
    @test count(==(VOCAB["cat"]), draws) > count(==(VOCAB["sofa"]), draws)
    # reproducible
    @test sample_negatives(uni, Xoshiro(5), V, (1, 2), 20) ==
          sample_negatives(uni, Xoshiro(5), V, (1, 2), 20)
end

@testset "one update: the documented step 5" begin
    m = toy_model()
    before = copy(wordvec(m, "cat"))
    info = update_pair!(m, "cat", "drinks", ["on", "wears", "queen"], 0.5)
    @test info.wants == [1.0, 0.0, 0.0, 0.0]
    @test all(info.probabilities .== 0.5)
    @test info.errors ≈ [0.25, -0.25, -0.25, -0.25]
    @test info.loss ≈ 4log(2)
    # W cannot move while C is zero
    @test iszero(info.ΔW)
    @test wordvec(m, "cat") == before
    # each changed C column is g times the center's column
    @test ctxvec(m, "drinks") ≈ 0.25 .* before
    @test ctxvec(m, "on") ≈ -0.25 .* before
    @test ctxvec(m, "queen") ≈ -0.25 .* before
    # and nothing else moved
    untouched = setdiff(1:22, [VOCAB[w] for w in ("drinks", "on", "wears", "queen")])
    @test all(iszero, m.C[:, untouched])
    # the new guess
    @test probability(m, "cat", "drinks") ≈ 0.500281 atol = 1e-6
    @test probability(m, "cat", "on") ≈ 0.499719 atol = 1e-6
    @test score(m, "cat", "drinks") ≈ 0.25 * sum(abs2, before)
end

@testset "one update: only five columns change" begin
    m = toy_model()
    # warm the tables up so that both W and C can move
    train!(m, PAIRS; epochs = 5, rng = Xoshiro(3))
    W0, C0 = copy(m.W), copy(m.C)
    info = update_pair!(m, "queen", "the", ["floor", "on", "crown"], 0.05)
    movedW = [j for j in 1:22 if m.W[:, j] != W0[:, j]]
    movedC = [j for j in 1:22 if m.C[:, j] != C0[:, j]]
    @test movedW == [VOCAB["queen"]]
    @test sort(movedC) == sort(changed_columns(info).C)
    @test length(movedC) == 4
    @test !iszero(info.ΔW)                      # now the center moves too
    # every ΔC column is a multiple of the center's (old) column: rank one
    w = W0[:, VOCAB["queen"]]
    for (k, j) in enumerate([VOCAB["the"], VOCAB["floor"], VOCAB["on"], VOCAB["crown"]])
        @test m.C[:, j] - C0[:, j] ≈ info.errors[k] .* w
    end
    # duplicated negatives are pushed twice, not once
    m2 = toy_model()
    train!(m2, PAIRS; epochs = 5, rng = Xoshiro(3))
    C2 = copy(m2.C)
    w2 = copy(wordvec(m2, "castle"))
    i2 = update_pair!(m2, "castle", "in", ["floor", "the", "floor"], 0.05)
    @test m2.C[:, VOCAB["floor"]] - C2[:, VOCAB["floor"]] ≈ (i2.errors[2] + i2.errors[4]) .* w2
end

@testset "the gradient is the derivative of the loss" begin
    # finite differences against the analytic update, on random vectors
    rng = Xoshiro(11)
    dim = 6
    v = Vocabulary(["a", "b", "c", "d"], Dict("a" => 1, "b" => 2, "c" => 3, "d" => 4), [4, 3, 2, 1])
    m = Model(v; dim = dim, negative = 2, rng = rng)
    m.C .= randn(rng, dim, 4) .* 0.3            # so that W can move as well
    m.W .= randn(rng, dim, 4) .* 0.3
    center, context, negs = 1, 2, [3, 4]

    L(W, C) = -log(sigmoid(dot(W[:, center], C[:, context]))) -
              sum(log(1 - sigmoid(dot(W[:, center], C[:, n]))) for n in negs)

    W0, C0 = copy(m.W), copy(m.C)
    info = update_pair!(m, center, context, negs, 1.0)   # lr = 1 so g is the raw error
    @test info.loss ≈ L(W0, C0)

    h = 1e-6
    for d in 1:dim
        Wp = copy(W0); Wp[d, center] += h
        numeric = (L(Wp, C0) - L(W0, C0)) / h
        @test -info.ΔW[d] ≈ numeric atol = 1e-5   # step is -gradient
    end
    for (k, j) in enumerate([context; negs])
        for d in 1:dim
            Cp = copy(C0); Cp[d, j] += h
            numeric = (L(W0, Cp) - L(W0, C0)) / h
            analytic = -info.errors[k] * W0[d, center]
            @test analytic ≈ numeric atol = 1e-5
        end
    end
end

@testset "learning rate schedule" begin
    @test learning_rate(1, 300) ≈ 0.051
    @test learning_rate(300, 300) ≈ 0.05 / 300 + 0.001 atol = 1e-12
    @test learning_rate(1, 300) > learning_rate(150, 300) > learning_rate(300, 300)
end

@testset "training the toy corpus" begin
    m = toy_model()
    tlog = train!(m, PAIRS; epochs = 300, snapshots = [0, 10, 300])
    @test length(tlog.losses) == 300
    @test tlog.losses[1] ≈ 4log(2) atol = 0.05        # first epoch is still coin-flipping
    @test tlog.losses[end] < 1.4                      # and it comes down
    @test minimum(tlog.losses) < tlog.losses[1]
    @test sort(collect(keys(tlog.snapshots))) == [0, 10, 300]
    @test tlog.snapshots[0].C == zeros(8, 22)         # before any update
    @test tlog.snapshots[300].W == m.W

    # the words that share company end up together
    @test similarity(m, "king", "queen") > 0.95      # identical neighbours
    @test similarity(m, "cat", "dog") > 0.6
    @test similarity(m, "milk", "water") > 0.6
    @test similarity(m, "cat", "king") < 0.5         # different kinds of sentence
    @test first(nearest_neighbours(m, "king", 1))[1] == "queen"
    @test "dog" in first.(nearest_neighbours(m, "cat", 3))

    # both tables have grown from their starting scale
    @test maximum(abs, m.W) > 1
    @test all(>(0), column_norms(m.C))

    # and the probabilities line up with what the counts predict
    N = cooccurrence(SENTENCES, VOCAB; window = 2)
    probs = sampling_probabilities(UniformNegatives(), 22)
    c = calibration(m, N, probs)
    @test c.correlation > 0.98
    @test c.mean_absolute_error < 0.05
    @test c.n == 22 * 21
end

@testset "training is reproducible and order matters" begin
    a = toy_model(); train!(a, PAIRS; epochs = 20, rng = Xoshiro(42))
    b = toy_model(); train!(b, PAIRS; epochs = 20, rng = Xoshiro(42))
    @test a.W == b.W && a.C == b.C
    c = toy_model(); train!(c, PAIRS; epochs = 20, rng = Xoshiro(43))
    @test a.W != c.W
    # a callback sees every epoch
    seen = Int[]
    train!(toy_model(), PAIRS; epochs = 4, callback = (e, _, _) -> push!(seen, e))
    @test seen == 1:4
end

@testset "target probabilities" begin
    N = cooccurrence(SENTENCES, VOCAB; window = 2)
    totals = row_totals(N)
    probs = sampling_probabilities(UniformNegatives(), 22)
    # 1 pull, 3 × 10 × 1/20 = 1.5 pushes per epoch
    @test target_probability(N, totals, probs, VOCAB["cat"], VOCAB["drinks"], 3) ≈ 0.4
    # twice the pulls, slightly fewer pushes
    @test target_probability(N, totals, probs, VOCAB["cat"], VOCAB["chases"], 3) ≈ 2 / 3.35 atol = 1e-3
    # a pair that never occurs has nothing pulling it up
    @test target_probability(N, totals, probs, VOCAB["cat"], VOCAB["crown"], 3) == 0
    T = target_matrix(N, probs, 3)
    @test size(T) == (22, 22)
    @test all(isnan, [T[i, i] for i in 1:22])
    @test T[VOCAB["cat"], VOCAB["drinks"]] ≈ 0.4
end

@testset "stepping" begin
    m = toy_model()
    s = TrainStepper(m, PAIRS; seed = 7)
    @test progress(s) == (update = 0, epoch = 0, pair = 0, of = 176)
    @test last_update(s) === nothing
    info = step!(s)
    @test progress(s).update == 1 && progress(s).epoch == 1 && progress(s).pair == 1
    @test last_update(s) === info
    @test info.lr ≈ learning_rate(1, 300)
    run_updates!(s, 9)
    @test progress(s).update == 10
    finish_epoch!(s)
    @test progress(s) == (update = 176, epoch = 1, pair = 176, of = 176)
    finish_epoch!(s)                                  # a whole further epoch
    @test progress(s).update == 352 && progress(s).epoch == 2

    # replaying from a reset gives exactly the same run
    tables = (copy(m.W), copy(m.C))
    reset!(s)
    @test m.C == zeros(8, 22)
    @test progress(s).update == 0
    run_updates!(s, 352)
    @test m.W ≈ tables[1] && m.C ≈ tables[2]
    @test_throws ArgumentError run_updates!(s, 0)

    # stepping and train! apply the same rule
    m2 = toy_model(); s2 = TrainStepper(m2, PAIRS; seed = 99)
    run_updates!(s2, 176)
    m3 = toy_model(); train!(m3, PAIRS; epochs = 1, rng = Xoshiro(99))
    @test m2.W ≈ m3.W && m2.C ≈ m3.C
end

@testset "analysis helpers" begin
    m = toy_model()
    train!(m, PAIRS; epochs = 60)
    S = similarity_matrix(m)
    @test size(S) == (22, 22)
    @test S ≈ S'
    @test all(≈(1), [S[i, i] for i in 1:22])
    @test all(-1.001 .<= S .<= 1.001)

    P, B, c = pca2(m.W)
    @test size(P) == (2, 22)
    @test size(B) == (8, 2)
    @test B' * B ≈ I(2)                     # orthonormal axes
    # the same basis projects another snapshot into the same frame
    P2, _, _ = pca2(m.W; basis = B, centre = c)
    @test P2 ≈ P

    r1 = neighbour_ranking(m, 3)
    @test length(r1) == 22 && all(length.(r1) .== 3)
    @test ranking_churn(r1, r1) == 0
    train!(m, PAIRS; epochs = 5)
    @test ranking_churn(r1, neighbour_ranking(m, 3)) >= 0

    @test cosine_similarity([1.0, 0], [1.0, 0]) ≈ 1
    @test cosine_similarity([1.0, 0], [0, 1.0]) ≈ 0 atol = 1e-12
    @test cosine_similarity([1.0, 0], [-1.0, 0]) ≈ -1
    @test length(analogy(m, "king", "queen", "cat")) == 3
    @test evaluate_loss(m, PAIRS) > 0
end

@testset "subsampling" begin
    keep = subsample_probabilities(VOCAB; threshold = 1e-3)
    @test length(keep) == 22
    @test all(0 .< keep .<= 1)
    # "the" is 26% of this corpus, so even a mild threshold thins it out
    @test keep[VOCAB["the"]] < keep[VOCAB["water"]]
    @test subsample_probabilities(VOCAB; threshold = 1.0) == ones(22)   # nothing dropped
    aggressive = subsample_probabilities(VOCAB; threshold = 1e-4)
    @test aggressive[VOCAB["the"]] < keep[VOCAB["the"]]
    ids = subsample(SENTENCES, VOCAB, aggressive, Xoshiro(4))
    @test all(s -> all(i -> 1 <= i <= 22, s), ids)
    @test sum(length, ids) < 62              # some tokens were dropped

    # a subsampled id stream can go straight into corpus_pairs
    p_ids = corpus_pairs(ids; window = 2)
    @test all(t -> 1 <= t[1] <= 22 && 1 <= t[2] <= 22, p_ids)
    @test length(p_ids) < length(PAIRS)
    # the two corpus_pairs methods agree when nothing is dropped
    full = [[VOCAB[w] for w in s] for s in SENTENCES]
    @test corpus_pairs(full; window = 2) == PAIRS
end

@testset "a second, larger corpus" begin
    # a corpus the toy defaults were not tuned for: different window, unigram
    # negatives, min_count, and only a handful of epochs
    text = repeat("red apple tastes sweet. green apple tastes sour. " *
                  "blue car drives fast. black car drives slow. ", 40)
    sents = tokenize(text)
    v = build_vocab(sents; min_count = 2)
    @test length(v) == 12
    pairs = corpus_pairs(sents, v; window = 5)
    m = Model(v; dim = 16, window = 5, negative = 5, rng = Xoshiro(2))
    sampler = UnigramNegatives(v; table_size = 10_000)
    slog = train!(m, pairs; epochs = 60, sampler = sampler, rng = Xoshiro(2))
    @test slog.losses[end] < slog.losses[1]
    # the two clusters share no words, so they must separate
    @test similarity(m, "apple", "tastes") > similarity(m, "apple", "car")
    @test similarity(m, "car", "drives") > similarity(m, "car", "apple")
end

@testset "real tokenizers" begin
    fixture = joinpath(@__DIR__, "fixtures", "tokenizers.json")
    if !isfile(fixture)
        @info "no fixtures — run experiments/make_fixtures.py to record them"
    else
        fx = JSON3.read(read(fixture, String))
        checked = 0
        for (name, spec) in pairs(fx)
            path = String(spec["path"])
            if !isfile(path)
                @info "skipping $name: its tokenizer.json is not on this machine"
                continue
            end
            checked += 1
            tok = load_hf_tokenizer(path; name = String(name))
            @test tok isa BPETokenizer
            @test vocab_size(tok) >= Int(spec["vocab_size"])

            @testset "$name" begin
                for c in spec["cases"]
                    text = String(c["text"])
                    want = Int.(c["ids"])
                    # ids match the Rust implementation exactly
                    @test encode(tok, text) == want
                    # so do the pieces, markers and all
                    @test token_strings(tok, text) == String.(c["tokens"])
                    # and decoding gets the text back
                    @test decode(tok, want) == String(c["decoded"])
                end
            end

            # ids and pieces are two views of one table
            for id in (0, 1, 100, vocab_size(tok) - 1)
                piece = token_string(tok, id)
                piece === nothing || @test token_id(tok, piece) == id
            end
            @test token_string(tok, vocab_size(tok)) === nothing
            @test token_id(tok, "\u0000 definitely not a token \u0000") === nothing

            # every text round-trips, including one that needs byte fallback
            for text in ("the queen wears a crown", "🐈 café", "x" ^ 50, "2026")
                @test decode(tok, encode(tok, text)) == text
            end
        end
        @test checked >= 1
    end

    @testset "rejects what it cannot do" begin
        mktempdir() do dir
            path = joinpath(dir, "unigram.json")
            write(path, """{"model": {"type": "Unigram", "vocab": []}}""")
            @test_throws ArgumentError load_hf_tokenizer(path)
        end
    end
end

@testset "safetensors" begin
    # build a small file by hand and read it back
    mktempdir() do dir
        path = joinpath(dir, "toy.safetensors")
        data = Float32[1, 2, 3, 4, 5, 6]                    # 2 rows × 3 cols, row-major
        header = """{"emb":{"dtype":"F32","shape":[2,3],"data_offsets":[0,$(sizeof(data))]}}"""
        open(path, "w") do io
            write(io, UInt64(ncodeunits(header)))
            write(io, header)
            write(io, data)
        end
        @test safetensors_names(path) == ["emb"]
        M = read_safetensor(path, "emb")
        @test size(M) == (3, 2)                              # transposed to column-major
        @test M[:, 1] == Float32[1, 2, 3]
        @test M[:, 2] == Float32[4, 5, 6]
        @test_throws KeyError read_safetensor(path, "nope")

        # bfloat16 is the top half of a float32
        vals = Float32[1.0, -2.5, 0.25, 64.0]
        bf = UInt16.(reinterpret(UInt32, vals) .>> 16)
        path2 = joinpath(dir, "bf.safetensors")
        header2 = """{"w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,$(sizeof(bf))]}}"""
        open(path2, "w") do io
            write(io, UInt64(ncodeunits(header2)))
            write(io, header2)
            write(io, bf)
        end
        @test vec(read_safetensor(path2, "w")) ≈ vals rtol = 0.01
    end
end

@testset "positional encoding" begin
    @testset "sinusoidal" begin
        P = sinusoidal_encoding(8, 16)
        @test size(P) == (8, 16)
        @test P[:, 1] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]   # position 0
        @test all(-1 .<= P .<= 1)
        @test_throws ArgumentError sinusoidal_encoding(7, 4)         # dim must be even
        # the definition, spot-checked
        dim, p, i = 8, 5, 2
        θ = p / 10_000.0^(2i / dim)
        @test P[2i + 1, p + 1] ≈ sin(θ)
        @test P[2i + 2, p + 1] ≈ cos(θ)
        # every position is distinct
        @test length(unique(eachcol(P))) == 16
        # the point of it: PE[:, p+k] is the same rotation of PE[:, p] for every p
        k = 3
        for pair in 0:(dim ÷ 2 - 1), p in (0, 4, 9)
            θk = k / 10_000.0^(2pair / dim)
            c, s_ = cos(θk), sin(θk)
            sinp, cosp = P[2pair + 1, p + 1], P[2pair + 2, p + 1]
            @test P[2pair + 1, p + k + 1] ≈ sinp * c + cosp * s_
            @test P[2pair + 2, p + k + 1] ≈ cosp * c - sinp * s_
        end
    end

    @testset "learned table" begin
        L = learned_positions(16, 32)
        @test size(L) == (16, 32)
        @test L == learned_positions(16, 32)                          # same seed
        @test L != learned_positions(16, 32; rng = Xoshiro(8))
        @test maximum(abs, L) < 0.2                                   # small init
    end

    @testset "RoPE" begin
        rng = Xoshiro(4)
        q, k = randn(rng, 16), randn(rng, 16)
        @test rope(q, 0) ≈ q                                          # no rotation at 0
        @test norm(rope(q, 137)) ≈ norm(q)                            # rotations preserve length
        # the property everything rests on: only the gap matters
        for (m, n) in ((7, 3), (100, 96), (5, 5), (2, 9))
            @test dot(rope(q, m), rope(k, n)) ≈ dot(rope(q, m - n), k)
        end
        # the matrix method rotates each column by its own position
        X = randn(rng, 8, 4)
        Y = rope(X, [0, 1, 2, 3])
        @test size(Y) == size(X)
        @test Y[:, 1] ≈ X[:, 1]
        @test Y[:, 3] ≈ rope(X[:, 3], 2)
        @test_throws DimensionMismatch rope(X, [0, 1])
        @test_throws ArgumentError rope(randn(rng, 7, 2), [0, 1])
        # a larger base turns the clocks more slowly
        @test dot(rope(q, 1; base = 1e6), q) > dot(rope(q, 1; base = 1e3), q)
    end

    @testset "ALiBi" begin
        s8 = alibi_slopes(8)
        @test length(s8) == 8
        @test s8 ≈ [2.0^-1, 2.0^-2, 2.0^-3, 2.0^-4, 2.0^-5, 2.0^-6, 2.0^-7, 2.0^-8]
        @test issorted(s8; rev = true) && all(>(0), s8)
        @test length(alibi_slopes(12)) == 12                          # not a power of two
        @test length(alibi_slopes(1)) == 1
        @test_throws ArgumentError alibi_slopes(0)

        B = alibi_bias(4, 6)
        @test size(B) == (6, 6, 4)
        @test all(B[i, i, h] == 0 for i in 1:6, h in 1:4)             # no penalty at zero distance
        @test all(B[i, j, h] == -Inf for i in 1:6, j in 1:6, h in 1:4 if j > i)
        @test B[6, 1, 1] < B[6, 5, 1]                                 # further is worse
        @test B[6, 1, 1] < B[6, 1, 4]                                 # steeper head, harsher penalty
        F = alibi_bias(2, 5; causal = false)
        @test F ≈ permutedims(F, (2, 1, 3))                           # symmetric without the mask
        @test all(isfinite, F)
    end

    @testset "what the encodings look like" begin
        P = sinusoidal_encoding(32, 40)
        S = position_similarity(P)
        @test size(S) == (40, 40)
        @test S ≈ S'
        @test all(≈(1), [S[i, i] for i in 1:40])
        @test S[1, 2] > S[1, 20]                                      # near beats far

        decay = rope_similarity(64, 12)
        @test length(decay) == 12
        @test decay[1] ≈ 1
        @test decay[2] < decay[1]
        @test mean(decay[8:12]) < mean(decay[1:4])                    # falls away with distance
    end
end

@testset "loading and saving vectors" begin
    m = toy_model()
    train!(m, PAIRS; epochs = 40)
    e = embedding(m; name = "toy")
    @test e isa Embedding && size(e) == (8, 22)
    @test similarity(e, "king", "queen") ≈ similarity(m, "king", "queen")
    @test first(nearest_neighbours(e, "king", 1)) == first(nearest_neighbours(m, "king", 1))

    mktempdir() do dir
        path = joinpath(dir, "toy.vec")
        save_vectors(path, e)
        back = load_vectors(path)
        @test length(back.vocab) == 22
        @test isapprox(back.W, e.W; atol = 1e-5)
        @test similarity(back, "king", "queen") ≈ similarity(e, "king", "queen") atol = 1e-4

        # a header-less file (the GloVe convention) loads the same way
        plain = joinpath(dir, "toy_noheader.vec")
        save_vectors(plain, e; header = false)
        @test length(load_vectors(plain).vocab) == 22
        # and max_words truncates from the top
        @test length(load_vectors(path; max_words = 5).vocab) == 5
        # an empty file is an error, not an empty embedding
        empty = joinpath(dir, "empty.vec"); write(empty, "")
        @test_throws ArgumentError load_vectors(empty)
    end
end

@testset "comparing two embeddings" begin
    a = embedding(let m = toy_model(); train!(m, PAIRS; epochs = 60, rng = Xoshiro(1)); m end)
    b = embedding(let m = toy_model(); train!(m, PAIRS; epochs = 60, rng = Xoshiro(2)); m end)

    @test length(shared_vocabulary(a, b)) == 22
    @test neighbour_overlap(a, a, VOCAB.words; k = 5).mean == 1.0          # with itself
    ov = neighbour_overlap(a, b, VOCAB.words; k = 5)
    @test 0 <= ov.mean <= 1
    @test length(ov.per_word) == 22
    @test ov.per_word["king"] >= 0                                          # queen is stable

    agree = similarity_agreement(a, b; pairs = 400)
    @test agree.n > 300
    @test -1 <= agree.correlation <= 1
    @test similarity_agreement(a, a; pairs = 200).correlation ≈ 1

    # two embeddings that share only some words
    half = Embedding(build_vocab([VOCAB.words[1:10]]), a.W[:, 1:10]; name = "half")
    @test length(shared_vocabulary(a, half)) == 10

    acc = analogy_accuracy(a)
    @test acc.asked == 0                       # the toy corpus has none of these words
    @test length(acc.skipped) == length(ANALOGY_QUESTIONS)
    toy_qs = [("cat", "dog", "king", "queen"), ("king", "queen", "cat", "dog")]
    acc2 = analogy_accuracy(a, toy_qs; k = 3)
    @test acc2.asked == 2 && 0 <= acc2.accuracy <= 1
    @test length(acc2.results) == 2
end

@testset "plotting extension" begin
    using CairoMakie                       # loading a backend activates the real methods
    m = toy_model()
    tlog = train!(m, PAIRS; epochs = 30, snapshots = [0, 10, 30])
    N = cooccurrence(SENTENCES, VOCAB; window = 2)
    groups = Dict(w => (w in ("the", "a") ? :other :
                        w in ("king", "queen", "kingdom", "rules", "wears", "crown",
                              "lives", "in", "castle") ? :royal : :animal)
                  for w in VOCAB.words)

    @test plot_tables(m) isa Figure
    @test plot_tables(m; values = false) isa Figure
    # the colour scale is opt-in, and asking for it widens the figure
    @test plot_tables(m; colorbar = true) isa Figure
    @test size(plot_tables(m; colorbar = true).scene)[1] >
          size(plot_tables(m).scene)[1]
    @test plot_similarity_matrix(m; colorbar = true) isa Figure
    @test plot_update(m, update_pair!(m, "cat", "drinks", ["on", "wears", "queen"], 0.05)) isa Figure
    @test plot_loss(tlog) isa Figure
    @test plot_similarity_matrix(m) isa Figure
    @test plot_similarity_matrix(m; order = sortperm(VOCAB.words), labels = false) isa Figure
    @test plot_cooccurrence(N, VOCAB) isa Figure
    @test plot_embedding_map(m; groups = groups) isa Figure
    @test plot_evolution(m, tlog.snapshots; groups = groups) isa Figure
    @test plot_positional_encoding(sinusoidal_encoding(32, 40)) isa Figure
    @test plot_positional_encoding(sinusoidal_encoding(16, 20); colorbar = true) isa Figure
    @test plot_position_decay(; dim = 32, len = 24, nheads = 3) isa Figure

    # a figure really does render
    mktempdir() do dir
        path = joinpath(dir, "tables.png")
        save(path, plot_tables(m))
        @test isfile(path) && filesize(path) > 1000
    end
end

end # top testset
