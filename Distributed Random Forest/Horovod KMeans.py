import tensorflow as tf
import numpy as np
import time as t
import horovod.tensorflow as hvd
#import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


np_points = np.loadtxt('points_np.txt', dtype=float)
num_points = len(np_points)
print('number of points', num_points)
dim = len(np_points[0])
num_cluster = 10




hvd.init()
rank = hvd.rank()
print('rank',rank)
points_per_device = int(num_points / hvd.size())

np_points = np_points[rank*points_per_device:rank*points_per_device + points_per_device-1]

#points = tf.constant(np_points, dtype=tf.float32)
points = tf.placeholder(dtype=tf.float32, name='global_sum_place_hold')


centroids = tf.get_variable(name='centroids', shape=[num_cluster, dim],
        initializer = tf.initializers.random_uniform(minval=0, maxval=10.0, seed=123))

bcast_result = hvd.broadcast(centroids, 0) 

init_centroids_sync = tf.assign(centroids, bcast_result)

expanded_points = tf.expand_dims(points, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(tf.square(tf.subtract(points, expanded_centroids)), 2)
assignments = tf.argmin(distances, 0)

loss_op = tf.reduce_sum(tf.reduce_min(distances, 0))
tf_sum = tf.unsorted_segment_sum(points, assignments, num_cluster)
tf_count = tf.unsorted_segment_sum(tf.ones_like(points), assignments, num_cluster)

loss_op_reduced = hvd.allreduce(loss_op, average=False)
tf.summary.scalar('loss_summary', loss_op_reduced)
tf_sum_reduced = hvd.allreduce(tf_sum, average=False)
tf_count_reduced = hvd.allreduce(tf_count, average=False)

new_centroids=tf_sum_reduced/tf_count_reduced

centroid_update =tf.assign(centroids, new_centroids)

merged_summary = tf.summary.merge_all()
print('rank', str(rank))
start_time = t.time()
#previous_centers = None
with tf.Session() as sess:
    if rank == 0:
        writer = tf.summary.FileWriter('horovod_model'+str(hvd.size())+'/', sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(init_centroids_sync)
    print('initial_centroids', sess.run(centroids))
    print('------------------------------')
    for i in range(100):
        centr, loss, merged_sum = sess.run([centroid_update, loss_op_reduced, merged_summary], feed_dict={points: np_points})
        #if previous_centers is not None:
        #    print('delta:', centr - previous_centers)
        #previous_centers = centr
        if rank == 0:
            writer.add_summary(merged_sum, i)
            print('step:',str(i+1),'loss:', loss)


computation_time = t.time() - start_time
print('computation time', str(computation_time), 'rank', rank)
print(centr)