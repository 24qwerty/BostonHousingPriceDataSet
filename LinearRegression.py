import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt

total_features,total_price=load_boston(True)
total_features.size
total_price.size

x=total_features[:450,5]
y=total_price[:450]
plt.scatter(x,y)

x=total_features[:450,7]
y=total_price[:450]
plt.scatter(x,y)

x=total_features[:450,12]
y=total_price[:450]
plt.scatter(x,y)

train_features=scale(total_features[:450,[5,7,12]])
train_price=total_price[:450]
test_features=scale(total_features[450:,[5,7,12]])
test_price=total_price[450:]

w=tf.Variable(tf.truncated_normal([3,1],mean=0.0,stddev=1.0,dtype=tf.float64))
b=tf.Variable(tf.zeros(1,dtype=tf.float64))

def calc(x, y):
# Returns predictions and error
    predictions = tf.add(b, tf.matmul(x, w))
    error = tf.reduce_mean(tf.square(y - predictions))
    return [ predictions, error ]

y, cost = calc(train_features, train_price)
learning_rate = 0.105
epochs = 3000
points = [[], []]

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:

    sess.run(init)

    for i in list(range(epochs)):

        sess.run(optimizer)

        if i % 10 == 0.:
            points[0].append(i+1)
            points[1].append(sess.run(cost))

        if i % 100 == 0:
            print(sess.run(cost))

    plt.plot(points[0], points[1], 'r--')
    plt.axis([0, epochs, 50, 600])
    plt.show()

    test_cost = calc(test_features, test_price)[1]
    print('Test error =', sess.run(test_cost), '\n')
