uv run python visualize_latent_tsne.py \
    --dataset dermamnist \
    --output outputs/tsne/dermamnist_latent_tsne.png \
    --per-class-samples 100 \
    --tsne-iter 5000 \
    --perplexity 15 \
    --early-exagg 24 \
    --learning-rate 300

uv run python visualize_latent_tsne.py \
    --dataset bloodmnist \
    --output outputs/tsne/bloodmnist_latent_tsne.png \
    --per-class-samples 100 \
    --tsne-iter 5000 \
    --perplexity 15 \
    --early-exagg 24 \
    --learning-rate 300

uv run python visualize_latent_tsne.py \
    --dataset aptos-2019-blindness-detection \
    --output outputs/tsne/aptos_latent_tsne.png \
    --per-class-samples 100 \
    --perplexity 10 \
    --early-exagg 24 \
    --learning-rate 300