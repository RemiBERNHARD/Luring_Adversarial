import itertools
import math
import numpy as np
import tensorflow as tf

from local_search_helper import LocalSearchHelper


class ParsimoniousAttack(object):
  """Parsimonious attack using local search algorithm"""

  def __init__(self, model, loss_func='xent', max_queries=20000, epsilon=0.03, batch_size=64, block_size=4, no_hier=False, 
               max_iters=1, **kwargs):

    """Initialize attack.
    
    Args:
      model: TensorFlow model
      args: arguments 
    """
    # Hyperparameter setting
    self.loss_func = loss_func
    self.max_queries = max_queries
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.block_size = block_size
    self.no_hier = no_hier
    self.max_iters = max_iters
         
    # Create helper
    self.local_search = LocalSearchHelper(model, self.epsilon, self.max_iters, self.loss_func)
 
  def _perturb_image(self, image, noise):
    """Given an image and a noise, generate a perturbed image. 
    
    Args:
      image: numpy array of size [1, 32, 32, 3], an original image
      noise: numpy array of size [1, 32, 32, 3], a noise

    Returns:
      adv_image: numpy array of size [1, 299, 299, 3], an adversarial image
    """
    adv_image = image + noise
    adv_image = np.clip(adv_image, 0, 1)
    return adv_image

  def _split_block(self, upper_left, lower_right, block_size):
    """Split an image into a set of blocks. 
    Note that a block consists of [upper_left, lower_right, channel]

    Args:
      upper_left: [x, y], the coordinate of the upper left of an image
      lower_right: [x, y], the coordinate of the lower right of an image
      block_size: int, the size of a block

    Return:
      blocks: list, a set of blocks
    """
    blocks = []
    xs = np.arange(upper_left[0], lower_right[0], block_size)
    ys = np.arange(upper_left[1], lower_right[1], block_size)
    for x, y in itertools.product(xs, ys):
      if (lower_right[0] == 32):
        for c in range(3):
          blocks.append([[x, y], [x+block_size, y+block_size], c])
      if (lower_right[0] == 28):
        blocks.append([[x, y], [x+block_size, y+block_size], 0]) 

    return blocks
  
  def perturb(self, image, label, index, sess):		
    """Perturb an image.
    
    Args:
      image: numpy array of size [1, 32, 32, 3], an original image
      label: numpy array of size [1], the label of the image (or target label)
      index: int, the index of the image
      sess: TensorFlow session

    Returns:
      adv_image: numpy array of size [1, 32, 32, 3], an adversarial image
      num_queries: int, the number of queries
      success: bool, True if attack is successful
    """
    # Set random seed by index for the reproducibility
    np.random.seed(index)
    
    # Local variables
    adv_image = np.copy(image)
    num_queries = 0
    block_size = self.block_size
    upper_left = [0, 0]
    lower_right = np.array(image.shape[1:3])
    
    # Split an image into a set of blocks
    blocks = self._split_block(upper_left, lower_right, block_size) 
    
    # Initialize a noise to -epsilon
    noise = -self.epsilon*np.ones_like(image, dtype=np.int32)

    # Construct a batch
    num_blocks = len(blocks)
    batch_size = self.batch_size if self.batch_size > 0 else num_blocks
    curr_order = np.random.permutation(num_blocks)

    # Main loop
    while True:
      # Run batch
      num_batches = int(math.ceil(num_blocks/batch_size))
      for i in range(num_batches):
        # Pick a mini-batch
        bstart = i*batch_size
        bend = min(bstart + batch_size, num_blocks)
        blocks_batch = [blocks[curr_order[idx]] for idx in range(bstart, bend)]
        # Run local search algorithm on the mini-batch
        noise, queries, loss, success = self.local_search.perturb(
          image, noise, label, sess, blocks_batch)
        num_queries += queries
        tf.logging.info("Block size: {}, batch: {}, loss: {:.4f}, num queries: {}".format(
          block_size, i, loss, num_queries))
        # If query count exceeds the maximum queries, then return False
        if num_queries > self.max_queries:
          return adv_image, num_queries, False
        # Generate an adversarial image
        adv_image = self._perturb_image(image, noise)
        # If attack succeeds, return True
        if success:
          return adv_image, num_queries, True
      
      # If block size is >= 2, then split the image into smaller blocks and reconstruct a batch
      if not self.no_hier and block_size >= 2:
        block_size //= 2
        blocks = self._split_block(upper_left, lower_right, block_size)
        num_blocks = len(blocks)
        batch_size = self.batch_size if self.batch_size > 0 else num_blocks
        curr_order = np.random.permutation(num_blocks)
      # Otherwise, shuffle the order of the batch
      else:
        curr_order = np.random.permutation(num_blocks)

