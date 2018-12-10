import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2


#在with块内部，session被设置成默认的session，使用完会自动关闭
with tf.Session() as sess:
    x.initializer.run()  # 等价于 tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval()  # 等价于 tf.get_default_session().run(f)
    print(result)