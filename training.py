import tensorflow as tf
import numpy as np
import datetime
from model_rnn import rnn_model
from tensorflow.python.framework import dtypes

from tensorflow.models.rnn.ptb import reader
from sklearn.metrics import confusion_matrix
from six.moves import cPickle
import math
import time
import os
import matplotlib.pyplot as plt


#Todo : dropout par layer, cell par layer


tf.flags.DEFINE_integer("state1_size", 100, "Number of hidden nodes for cell 1 (default: 100)")
tf.flags.DEFINE_integer("state2_size", 100, "Number of hidden nodes for cell 2 (default: 100)")
tf.flags.DEFINE_integer("state3_size", 100, "Number of hidden nodes for cell 3 (default: 100)")
tf.flags.DEFINE_float("keep_prob_layer1", 0.9, "Probability to keep nodes in layer 1 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer2", 0.8, "Probability to keep nodes in layer 2 (default: 0.8)")
tf.flags.DEFINE_float("keep_prob_layer3", 0.7, "Probability to keep nodes in layer 3 (default: 0.6)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("nb_epochs", 10, "Number of training step (default: 2)")
tf.flags.DEFINE_integer("eval_every", 100, "Number of steps between every eval print (default: 100)")
tf.flags.DEFINE_float("learning_rate", 0.05, "Initial learning rate (default: 0.0005)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.00000, "L2 regularizaion lambda (default: 0.0001)")
tf.flags.DEFINE_integer("num_steps", 10, "Num steps (default: 10)")
tf.flags.DEFINE_integer("num_classes", 2, "Num classes (default: 2)")
tf.flags.DEFINE_boolean("tf_", True, "Use tensorflow (default: True)")
tf.flags.DEFINE_boolean("verbose", True, "Use verbose (default: True)")
tf.flags.DEFINE_integer("num_layers", 1, "Num layers (default: 1)")
tf.flags.DEFINE_string("cell_type", 'BASIC', "Cell type (default: 'BASIC')")
tf.flags.DEFINE_string("is_training", 'FALSE', "us dropout or not")
tf.flags.DEFINE_string("save_dir", 'save', "Where to save models / params (default: 'save')")

raw_data = open('tiny-shakespeare.txt', 'r').read() # should be simple plain text file
vocab = set(raw_data)
vocab_size = len(vocab)

idx_to_vocab = dict(enumerate(vocab))
vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
data = [vocab_to_idx[c] for c in raw_data]
del raw_data

num_c = vocab_size


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(data, batch_size, num_steps)


with open(os.path.join(FLAGS.save_dir, 'config.pkl'), 'wb') as f:
    cPickle.dump(FLAGS, f)

with tf.Graph().as_default():
    with tf.Session() as sess:

        sqrt_0 = math.sqrt(1.0 / float(FLAGS.batch_size))
        sqrt_1 = math.sqrt(1.0 / float(FLAGS.state1_size + FLAGS.num_classes))
        sqrt_2 = math.sqrt(1.0 / float(FLAGS.state1_size + FLAGS.num_classes))
        sqrt_3 = math.sqrt(1.0 / float(FLAGS.state1_size + FLAGS.num_classes))
        list_stddev = [sqrt_0, sqrt_1, sqrt_2, sqrt_3]
        current_list = [sqrt for sqrt in list_stddev[:FLAGS.num_layers]]

        rnn = rnn_model(batch_size = FLAGS.batch_size,
                        state_size = FLAGS.state1_size,
                        num_steps = FLAGS.num_steps,
                        num_classes = num_c,
                        stddev_init = [sqrt_0, sqrt_1],
                        tf_ = FLAGS.tf_,
                        num_layers = FLAGS.num_layers,
                        cell_type = FLAGS.cell_type,
                        data_size = len(data),
                        is_training = FLAGS.is_training)
        
        training_losses = []

        global_step = tf.Variable(0)
        init_lr = FLAGS.learning_rate
        lr = tf.train.exponential_decay(init_lr, global_step, 500, 0.90, staircase=True)
        optimizer = tf.train.AdagradOptimizer(init_lr)
        train_step = optimizer.minimize(rnn.loss)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.scalar_summary("loss", rnn.loss)
        train_summary_op =  loss_summary
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())

        start_t = time.time()

        for idx, epoch in enumerate(gen_epochs(FLAGS.nb_epochs, FLAGS.num_steps, FLAGS.batch_size)):
            training_loss = 0
            t = time.time()
            training_state = rnn.initial_state.eval()
            if FLAGS.verbose:
                print("\nEPOCH", idx)
                nb_steps_by_epoch = (len(data) // (FLAGS.batch_size * FLAGS.num_steps))
                print("nb steps by epochs", nb_steps_by_epoch )

            for step, (X, Y) in enumerate(epoch):
	            
                summaries, training_loss_, training_state, _ = \
                    sess.run([train_summary_op, rnn.loss,
                              rnn.final_state,
                              train_step],
                                  feed_dict={rnn.x:X, rnn.y:Y, rnn.initial_state:training_state})
                train_summary_writer.add_summary(summaries, step)
                training_loss += training_loss_

                if step % FLAGS.eval_every == 0 and step > 0:
                    checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')

                    if FLAGS.verbose:
                    	estimated_training_time = (nb_steps_by_epoch + FLAGS.eval_every)* FLAGS.nb_epochs * (time.time() - t) / FLAGS.eval_every
                    	elapsed_time = time.time() - start_t
                    	print("Average loss at step %d : %.5f" % (step, training_loss / FLAGS.eval_every),
                              "Last %i steps took %.1f s" % (FLAGS.eval_every, (time.time() - t)),
                              "Approximative remaining train time : %s" % time.strftime("%H:%M:%S", time.gmtime((estimated_training_time - elapsed_time))))
                        t = time.time()
                    training_losses.append(training_loss/FLAGS.eval_every)
                    training_loss = 0
                
            print "end epoch %d. Will soon add test on valid dataset at this point" % idx
            saver.save(sess, checkpoint_path, global_step = global_step)
            print "training with learning rate : %.5f" % lr.eval()
            print("model saved to {}".format(checkpoint_path))
        plt.plot(training_losses)
        plt.show()
