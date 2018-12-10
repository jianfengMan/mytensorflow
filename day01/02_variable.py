# 进入一个交互式 TensorFlow 会话
import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器initializer op的run()方法初始化'x'
x.initializer.run()

# 增加一个减法 subtract op, 从 'x' 减去 'a'. 运行减法 op, 输出结果
sub = tf.subtract(x, a)
print(sub.eval())

# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

# 启动图
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))

    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# 代码中 assign() 操作是图所描绘的表达式的一部分, 正如 add() 操作一样. 所以在调用 run() 执行表达式之前, 它并不会真正执行赋值操作.
