import sys

sys.path.append("models")
from model import define_gan, define_discriminator, define_generator, train
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    generator = define_generator()
    discriminator = define_discriminator()
    gan = define_gan(generator, discriminator, (256, 256, 3))
    train(discriminator, generator, gan)
