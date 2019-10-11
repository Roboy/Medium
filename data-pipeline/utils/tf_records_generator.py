import numpy as np
import tensorflow as tf


# Helperfunctions to make feature definition more readable
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def array_to_bytes(arr):
    import zlib
    arr_shape = arr.shape
    arr_bytes = np.real(arr).flatten().astype(np.float32).tobytes()
    # return arr_bytes
    return arr_shape, zlib.compress(arr_bytes)


# TS3 features
def generate_ts3_feature(ts3_points, heatmaps, filename=None):
    heatmaps_shape, heatmaps_compressed = array_to_bytes(heatmaps)

    feature_dict = {
        'filename': _bytes_feature(bytes(filename, 'utf-8')),
        'heatmaps': _bytes_feature(heatmaps_compressed),
        'heatmaps/height': _int64_feature(heatmaps_shape[0]),
        'heatmaps/width': _int64_feature(heatmaps_shape[1]),
        'heatmaps/depth': _int64_feature(heatmaps_shape[2])
    }

    for i, ts3 in enumerate(ts3_points):
        feature_dict['ts3/{}/x'.format(i)] = _float_feature([point['x'] for point in ts3])
        feature_dict['ts3/{}/y'.format(i)] = _float_feature([point['x'] for point in ts3])
        feature_dict['ts3/{}/z'.format(i)] = _float_feature([point['x'] for point in ts3])
        feature_dict['ts3/{}/intensity'.format(i)] = _float_feature([point['x'] for point in ts3])
        feature_dict['ts3/{}/num_points'.format(i)] = _int64_feature(len(ts3))

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


# P2G features
def generate_p2g_feature(X_rdm, heatmaps, filename=None):

    X_rdm_shape, X_rdm_compressed = array_to_bytes(X_rdm)
    heatmaps_shape, heatmaps_compressed = array_to_bytes(heatmaps)

    feature_dict = {
        'filename': _bytes_feature(bytes(filename, 'utf-8')),

        'heatmaps': _bytes_feature(heatmaps_compressed),
        'heatmaps/height': _int64_feature(heatmaps_shape[0]),
        'heatmaps/width': _int64_feature(heatmaps_shape[1]),
        'heatmaps/depth': _int64_feature(heatmaps_shape[2]),

        'rdm': _bytes_feature(X_rdm_compressed),
        'rdm/height': _int64_feature(X_rdm_shape[0]),
        'rdm/width': _int64_feature(X_rdm_shape[1]),
        'rdm/depth': _int64_feature(X_rdm_shape[2]),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def parse_p2g_example(serialized_example):
    p2g_feature = {
        'filename': tf.FixedLenFeature([], tf.string),

        'heatmaps': tf.FixedLenFeature([], tf.string),
        'heatmaps/height': tf.FixedLenFeature([], tf.int64),
        'heatmaps/width': tf.FixedLenFeature([], tf.int64),
        'heatmaps/depth': tf.FixedLenFeature([], tf.int64),

        'rdm': tf.FixedLenFeature([], tf.string),
        'rdm/height': tf.FixedLenFeature([], tf.int64),
        'rdm/width': tf.FixedLenFeature([], tf.int64),
        'rdm/depth': tf.FixedLenFeature([], tf.int64),
    }

    example = tf.parse_single_example(serialized_example, p2g_feature)

    heatmaps_str = tf.io.decode_compressed(example['heatmaps'], compression_type='ZLIB')
    heatmaps_flat = tf.decode_raw(heatmaps_str, out_type=tf.float32)
    heatmaps_flat = tf.cast(heatmaps_flat, tf.float32)
    heatmaps = tf.reshape(heatmaps_flat, tf.stack([64, 64, 13]))  # Hardcore shape because dynamic loading is a b***h

    rdm_str = tf.io.decode_compressed(example['rdm'], compression_type='ZLIB')
    rdm_flat = tf.decode_raw(rdm_str, out_type=tf.float32)
    rdm_flat = tf.cast(rdm_flat, tf.float32)
    rdm = tf.reshape(rdm_flat, tf.stack([64, 64, 8]))  # Hardcore shape because dynamic loading is a b***h

    # raw_str = tf.io.decode_compressed(example['raw'], compression_type='ZLIB')
    # raw_flat = tf.decode_raw(raw_str, out_type=tf.uint8)
    # raw_flat = tf.cast(raw_flat, tf.float32)
    # raw = tf.reshape(raw_flat, tf.stack([128, 64, 8]))  # Hardcore shape because dynamic loading is a b***h

    return rdm, heatmaps # raw, rdm, heatmaps

# Walabot
def generate_wlb_feature(X_wlb, heatmaps, filename=None):

    X_wlb_shape, X_wlb_compressed = array_to_bytes(X_wlb)
    heatmaps_shape, heatmaps_compressed = array_to_bytes(heatmaps)

    feature_dict = {
        'filename': _bytes_feature(bytes('test', 'utf-8')),

        'heatmaps': _bytes_feature(heatmaps_compressed),
        'heatmaps/height': _int64_feature(heatmaps_shape[0]),
        'heatmaps/width': _int64_feature(heatmaps_shape[1]),
        'heatmaps/depth': _int64_feature(heatmaps_shape[2]),

        'wlb': _bytes_feature(X_wlb_compressed),
        'wlb/height': _int64_feature(X_wlb_shape[0]),
        'wlb/width': _int64_feature(X_wlb_shape[1]),
        'wlb/depth': _int64_feature(X_wlb_shape[2]),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))

def parse_wlb_example(serialized_example):
    wlb_feature = {
        'filename': tf.FixedLenFeature([], tf.string),

        'heatmaps': tf.FixedLenFeature([], tf.string),
        'heatmaps/height': tf.FixedLenFeature([], tf.int64),
        'heatmaps/width': tf.FixedLenFeature([], tf.int64),
        'heatmaps/depth': tf.FixedLenFeature([], tf.int64),

        'wlb': tf.FixedLenFeature([], tf.string),
        'wlb/height': tf.FixedLenFeature([], tf.int64),
        'wlb/width': tf.FixedLenFeature([], tf.int64),
        'wlb/depth': tf.FixedLenFeature([], tf.int64),
    }

    example = tf.parse_single_example(serialized_example, wlb_feature)

    heatmaps_str = tf.io.decode_compressed(example['heatmaps'], compression_type='ZLIB')
    heatmaps_flat = tf.decode_raw(heatmaps_str, out_type=tf.uint8)
    heatmaps_flat = tf.cast(heatmaps_flat, tf.float32)
    heatmaps = tf.reshape(heatmaps_flat,
                          tf.stack([128, 128, 13]))  # Hardcore shape because dynamic loading is a b***h

    wlb_str = tf.io.decode_compressed(example['wlb'], compression_type='ZLIB')
    wlb_flat = tf.decode_raw(wlb_str, out_type=tf.uint8)
    wlb_flat = tf.cast(wlb_flat, tf.float32)
    wlb = tf.reshape(wlb_flat, tf.stack([116, 25, 8]))  # Hardcore shape because dynamic loading is a b***h

    return wlb, heatmaps