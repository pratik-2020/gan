class NumGAN:

  def _init_(self):
    self.BUFFER_SIZE = 60000
    self.BATCH_SIZE = 256
    self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    self.gen_mod = self.build_gen_mod()
    self.disc_mod = self.build_disc_mod()
    self.train_dataset = self.gen_dt()
  
  def gen_dt(self):
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    BUFFER_SIZE = 60000
    BATCH_SIZE = 256
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

    return train_dataset
  
  def build_gen_mod(self):
    model = tf.keras.Sequential()
    model.add(Dense(512, activation="leaky_relu", input_shape=(200,)))
    model.add(BatchNormalization())
    model.add(Dense(512, activation="leaky_relu"))
    model.add(BatchNormalization())
    model.add(Dense(512, activation="leaky_relu"))
    model.add(BatchNormalization())
    model.add(Dense(512, activation="leaky_relu"))
    model.add(BatchNormalization())
    model.add(Dense(512, activation="leaky_relu"))
    model.add(BatchNormalization())
    model.add(Dense(784, activation="leaky_relu"))
    model.add(BatchNormalization())
    model.add(Reshape((28,28,1)))

    return model
  
  def build_disc_mod(self):
    model = tf.keras.Sequential()
    model.add(Conv2D(16, (3,3), activation="relu"))
    model.add(Conv2D(8, (1,1), activation="relu"))
    model.add(Conv2D(16, (5,5), activation="relu"))
    model.add(Conv2D(8, (1,1), activation="relu"))
    model.add(Conv2D(16, (3,3), activation="relu"))
    model.add(Conv2D(8, (1,1), activation="relu"))
    model.add(Conv2D(16, (5,5), activation="relu"))
    model.add(Conv2D(8, (1,1), activation="relu"))
    model.add(Conv2D(16, (3,3), activation="relu"))
    model.add(Conv2D(8, (1,1), activation="relu"))
    model.add(Conv2D(16, (5,5), activation="relu"))
    model.add(Conv2D(8, (1,1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(216, activation="relu"))
    model.add(Dense(216, activation="relu"))
    model.add(Dense(216, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model
  def discriminator_loss(self, real_output, fake_output):
    real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

  def generator_loss(self, fake_output):
      return self.cross_entropy(tf.ones_like(fake_output), fake_output)
  
  @tf.function
  def train_step(self, images):
      noise = tf.random.normal([self.BATCH_SIZE, 200])

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          generated_images = self.gen_mod(noise, training=True)

          real_output = self.disc_mod(images, training=True)
          fake_output = self.disc_mod(generated_images, training=True)

          gen_loss = self.generator_loss(fake_output)
          disc_loss = self.discriminator_loss(real_output, fake_output)

      gradients_of_generator = gen_tape.gradient(gen_loss, self.gen_mod.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.disc_mod.trainable_variables)

      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.gen_mod.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc_mod.trainable_variables))