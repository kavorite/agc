""" An implementation of Adaptive Gradient Clipping
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
Code references:
  * Official JAX implementation (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
  * Ross Wightman's implementation https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
"""

import tensorflow as tf


def compute_norm(x, axis, keepdims):
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def unitwise_norm(x):
    rank = tf.rank(x)
    keepdims = rank >= 1
    axis = tf.cond(rank >= 4, lambda: [0, 1, 2], lambda: 0)
    return compute_norm(x, keepdims=keepdims, axis=axis)


def adaptive_clip_gradients(parameters, gradients, clip_factor=0.01, eps=1e-3):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
        p_norm = unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, eps) * clip_factor
        grad_norm = unitwise_norm(grads)
        clipped_grad = grads * tf.math.divide_no_nan(
            max_norm, tf.math.maximum(grad_norm, eps)
        )
        new_grad = tf.where(grad_norm < max_norm, grads, clipped_grad)
        new_grads.append(new_grad)
    return new_grads


def extend_with_agc(cls):
    class AGCOptimizer(cls):
        def __init__(self, *args, clip_factor=1e-2, eps=1e-3, **kwargs):
            super().__init__(*args, **kwargs)
            self.clip_factor = clip_factor
            self.eps = eps

        def get_config(self):
            base = super().get_config()
            conf = dict(clip_factor=self.clip_factor, eps=self.eps)
            base.update(conf)
            return base

        def apply_gradients(self, gradvars, **kwargs):
            grads, vars = zip(*gradvars)
            grads = adaptive_clip_gradients(
                grads, vars, clip_factor=self.clip_factor, eps=self.eps
            )
            return super().apply_gradients(zip(grads, vars), **kwargs)

    return AGCOptimizer


AGCOptimizer = extend_with_agc(tf.keras.optimizers.Optimizer)
