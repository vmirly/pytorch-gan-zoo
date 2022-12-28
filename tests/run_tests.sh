# PyTorch GAN Zoo

# Running FC train/eval
for loss in vanilla wgan wgan-gp wgan-lp ; do
    python -m run.basic_gan.train_fc \
	--num_epochs=1  --z_dim=10 \
	--num_hidden_units=16  \
	--network_type=fc-large  \
	--dataset_name=fashion-mnist \
	--loss_type=${loss};
done
