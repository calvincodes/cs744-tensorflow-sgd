from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time
# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.DEBUG)
clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
        ]
    })
clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
        ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222"
        ]
    })
clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
        ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222",
        "10.10.1.3:2222",
        ]
    })
clusterSpec = {
        "single": clusterSpec_single,
        "cluster": clusterSpec_cluster,
        "cluster2": clusterSpec_cluster2
        }
clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
mnist = input_data.read_data_sets("data/", one_hot=True)
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    #put your code here
    is_chief = (FLAGS.task_index == 0)
    # Graph
    worker_device = worker_device="/job:worker/task:%d" % FLAGS.task_index
    ps_device="/job:ps/cpu:0"
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device,\
            ps_device=ps_device,\
            cluster=clusterinfo)):
        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        # Construct model
        pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Gradient Descent
        opt= tf.train.GradientDescentOptimizer(learning_rate)
        opt= tf.train.SyncReplicasOptimizer(opt, \
                replicas_to_aggregate=3, total_num_replicas=3)
        training_opt = opt.minimize(loss, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    local_init_op = opt.local_step_init_op
    if is_chief:
        local_init_op = opt.chief_init_op
    ready_for_local_init_op = opt.ready_for_local_init_op
    chief_queue_runner = opt.get_chief_queue_runner()
    init_token_op = opt.get_init_tokens_op()
    # Minimize error using cross entropy
    init_op = tf.initialize_all_variables()
    sv = tf.train.Supervisor(is_chief=is_chief, init_op=init_op,\
            local_init_op=local_init_op, \
            ready_for_local_init_op=ready_for_local_init_op,\
            recovery_wait_secs=1, \
            global_step=global_step)
    sess = sv.prepare_or_wait_for_session(server.target)
    if is_chief:
        sess.run(init_token_op)
        sv.start_queue_runners(sess, [chief_queue_runner])
    time_begin = time.time()
    local_step = 0
    train_steps=6000
    while True:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_feed = {x: batch_xs, y: batch_ys}
        _, step = sess.run([training_opt, global_step], feed_dict=train_feed)
        local_step += 1
        with sess.as_default():
            if (step % 250)==0 :
                print("Accuracy:", \
                        accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        if step >= train_steps:
            break
time_end = time.time()
training_time = time_end - time_begin
print("Training elapsed time: %f s" % training_time)