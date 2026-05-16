uv run python visualize_latent_tsne.py \
    --dataset dermamnist \
    --output dermamnist_latent_tsne.png \
    --max-samples 3000

uv run python visualize_latent_tsne.py \
    --dataset bloodmnist \
    --output bloodmnist_latent_tsne.png \
    --max-samples 3000

uv run python visualize_latent_tsne.py \
    --dataset aptos-2019-blindness-detection \
    --output aptos_latent_tsne.png \
    --max-samples 3000