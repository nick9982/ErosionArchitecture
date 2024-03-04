import sys
sys.path.append('data')
sys.path.append('models')
from erosionData import ErosionData
from model import define_gan, define_discriminator, define_generator, train
import numpy as np

if __name__ == "__main__":
    generator = define_generator()
    discriminator = define_discriminator()
    gan = define_gan(generator, discriminator, (256, 256, 3))
    train(discriminator, generator, gan)
