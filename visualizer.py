from pathlib import Path
import imageio
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, filenames, image_size=28, batch_size=64, create_gif=True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.create_gif = create_gif
        self.filenames = filenames
        self.image_dir = Path('../images/gan-sample-images')

    def sample_images(self, images, epoch):
        images = np.reshape(images, (self.batch_size, self.image_size, self.image_size))

        fig = plt.figure(figsize=(4, 4))

        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        if self.create_gif:
            current_epoch_filename = self.image_dir.joinpath(f"gan_epoch_{epoch}.png")
            self.filenames.append(current_epoch_filename)
            plt.savefig(current_epoch_filename)

        plt.close()

    def generate_gif(self):
        images = []
        for filename in self.filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave("../images/gan.gif", images)
