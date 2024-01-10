from symnn.de_learn_network import log_activation, \
    eql_model_v2, add_ln_block, set_model_l1_l2, L1L2_m, \
    eql_model_v3, add_depth_ln_layer, add_width_ln_block, \
    eql_opt, copy_tf_model
from symnn.utils import eq_complexity, get_sympy_expr_v2
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


def preprocess_data(train_x, train_y=None, is_train=True, val_split=0.2):
    if is_train:
        train_count = int(train_x[0].shape[0] * (1 - val_split))
        val_x = train_x[:, train_count:]
        val_x = [x_in for x_in in val_x]
        train_x = train_x[:, :train_count]
        train_x = [x_in for x_in in train_x]
        val_y = train_y[train_count:]
        val_y = [val_y]
        train_y = train_y[:train_count]
        train_y = [train_y]
        return train_x, val_x, train_y, val_y
    else:
        train_x = [x_in for x_in in train_x]
        return train_x


def reg_stages_train(model, train_x, train_y, num_epochs, reg_change=0.3, l1_reg=1e-3, l2_reg=1e-3):
    stage1_epochs = int(reg_change * num_epochs)
    stage2_epochs = num_epochs - stage1_epochs
    train_x, val_x, train_y, val_y = preprocess_data(train_x, train_y)
    set_model_l1_l2(model, l1=l1_reg, l2=l2_reg)
    hist1 = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                      epochs=stage1_epochs, batch_size=32)
    print('intermediate weights', model.get_weights())
    set_model_l1_l2(model, l1=0, l2=0)
    hist2 = model.fit(train_x, train_y, validation_data=(val_x, val_y),
                      epochs=stage2_epochs, batch_size=32)
    pred_y = model.predict(val_x)
    print(val_y, pred_y)
    mse = mean_squared_error(val_y[0], pred_y)
    return model, [hist1, hist2], mse


def train_model_reg_stage(train_x, train_y, ln_block_count, num_epochs, decay_steps=1000, reg_change=0.3):
    print(train_x.shape)
    train_count = train_x.shape[1]
    x_dim = train_x.shape[0]
    reg_change1 = reg_change
    reg_change2 = 1
    # weight_thresh = 0.01
    stage1_epochs = int(reg_change1 * num_epochs)
    stage2_epochs = int(reg_change2 * num_epochs) - stage1_epochs
    # stage3_epochs = num_epochs - stage2_epochs
    model = eql_model_v2(input_size=x_dim, ln_block_count=ln_block_count,
                         decay_steps=train_count / 10)
    model, history = reg_stages_train(model, train_x, train_y, num_epochs, reg_change)
    return model, history


def train_model_growth(train_x, train_y, start_ln_block, num_epochs, growth_steps=2,
                       init_lr=0.01, decay_steps=1000,
                       reg_change=0.3, l1_reg=1e-2, l2_reg=1e-2):
    # train_x = [np.array(x) for x in np.array(train_x.transpose())]
    print(train_x)
    x_dim = len(train_x)
    cur_ln_blocks = start_ln_block
    model = eql_model_v2(x_dim, ln_block_count=start_ln_block, decay_steps=decay_steps)
    train_history = []
    print('model weights', model.get_weights())
    opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
    # train_history.append(
    #     model.fit(train_x, train_y, epochs=num_epochs, batch_size=32, validation_split=0.2)
    # )

    model, history, mse = reg_stages_train(model, train_x, train_y, num_epochs, reg_change=reg_change,
                                           l1_reg=l1_reg, l2_reg=l2_reg)
    mse_cur = mse
    print('MSE', mse)
    actual_blocks = start_ln_block
    train_history += history
    for i in range(growth_steps):
        print(model.get_weights())
        tf.keras.utils.get_custom_objects().update({'log_activation': log_activation,
                                                    'L1L2_m': L1L2_m})
        model_new = tf.keras.models.clone_model(model)
        model_new.set_weights(model.get_weights())
        model_new = add_ln_block(x_dim, model_new, cur_ln_blocks)
        print(model_new.get_weights())
        cur_ln_blocks += 1
        opt = eql_opt(decay_steps=decay_steps, init_lr=init_lr)
        model_new.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_squared_error'])
        # train_history.append(
        #     model.fit(train_x, train_y, epochs=num_epochs, batch_size=32, validation_split=0.2)
        # )
        model_new, history, mse = reg_stages_train(model_new, train_x, train_y, num_epochs,
                                                   l1_reg=l1_reg, l2_reg=l2_reg,
                                                   reg_change=reg_change)
        print('MSE', mse)
        if mse > mse_cur * 0.8 and mse_cur < 1e-4:
            break
        model = tf.keras.models.clone_model(model_new)
        model.set_weights(model_new.get_weights())
        actual_blocks += 1
        mse_cur = mse
        train_history += history
    print(model.get_weights())
    return model, train_history, actual_blocks, mse_cur


def train_model_iter(cur_model, train_x, train_y, num_epochs, l1_reg, l2_reg):
    set_model_l1_l2(cur_model, l1=l1_reg, l2=l2_reg)
    train_x, val_x, train_y, val_y = preprocess_data(train_x, train_y)
    hist = cur_model.fit(train_x, train_y, validation_data=(val_x, val_y),
                         epochs=num_epochs, batch_size=32)
    print(cur_model.get_weights())
    pred_y = cur_model.predict(val_x)
    mse = mean_squared_error(val_y[0], pred_y)
    return cur_model, hist, mse


def train_model_growth_v3(train_x, train_y, max_ln_blocks, max_line_blocks,
                          num_epochs, decay_steps=1000, l1_reg=1e-2, l2_reg=1e-2,
                          early_stop_thresh=0.95):
    if len(max_line_blocks) != len(max_ln_blocks):
        raise ValueError("max_ln_blocks should have same length as max_line_blocks")
    if max_line_blocks[-1] != 1:
        raise ValueError("last dense layer should have size 1")

    x_dim = len(train_x)
    cur_ln_blocks = [2, 1]
    cur_line_blocks = [1, 1]
    model = eql_model_v3(x_dim, ln_blocks=cur_ln_blocks, lin_blocks=cur_line_blocks,
                         opt=eql_opt(decay_steps), compile=True)
    model, hist, mse = train_model_iter(model, train_x, train_y, num_epochs, l1_reg, l2_reg)
    mse_cur = mse
    train_history = [hist]
    early_stop_flag = False
    for depth_idx in range(len(max_ln_blocks)):
        if early_stop_flag:
            break
        if len(cur_ln_blocks) <= depth_idx:
            cur_ln_blocks.append(0)
            new_line_blocks = cur_line_blocks[:depth_idx - 1] + [max_line_blocks[depth_idx - 1]] + \
                              [cur_line_blocks[depth_idx - 1]]
        else:
            new_line_blocks = cur_line_blocks.copy()
        for i in range(max_ln_blocks[depth_idx]):
            if cur_ln_blocks[depth_idx] > i:
                continue
            model_new = copy_tf_model(model)
            if cur_ln_blocks[depth_idx] == 0:
                print('adding depth')
                print(cur_ln_blocks, new_line_blocks)
                model_new = add_depth_ln_layer(x_dim, model_new, cur_ln_blocks[:-1], new_line_blocks,
                                               opt=eql_opt(decay_steps))
            else:
                print('adding width')
                print('ln blocks', cur_ln_blocks)
                model_new = add_width_ln_block(x_dim, model_new, cur_ln_blocks, cur_line_blocks,
                                               opt=eql_opt(decay_steps))

            print(model_new.get_weights())
            model_new, hist, mse = train_model_iter(model_new, train_x, train_y,
                                                    num_epochs, l1_reg, l2_reg)
            print('new model summary')
            print(model_new.summary())
            print(model_new.get_weights())
            print('MSE', mse, mse_cur)
            if mse > mse_cur * early_stop_thresh:
                early_stop_flag = True
                break
            cur_ln_blocks[depth_idx] += 1
            print('current ln blocks', cur_ln_blocks)
            cur_line_blocks = new_line_blocks.copy()
            model = copy_tf_model(model_new)
            mse_cur = mse
            train_history += [hist]
    if cur_ln_blocks[-1] == 0:
        cur_ln_blocks.pop()
    print(model.get_weights())
    return model, train_history, cur_ln_blocks, cur_line_blocks


def select_best_model(train_x, train_y, start_ln_block, num_epochs, growth_steps=2,
                      init_lr=0.01, decay_steps=1000,
                      reg_change=0.3, l1_reg=1e-2, l2_reg=1e-2, num_iter=3,
                      round_digits=3, complexity_weight=1e-6):
    best_mse = float('inf')
    best_err = float('inf')
    best_model = None
    best_train_hist = None
    best_blk = None
    best_eq = None
    model_eq_list = []
    for i in range(num_iter):
        model, train_history, actual_blocks, mse_cur = train_model_growth(
            train_x, train_y, start_ln_block, num_epochs, growth_steps,
            init_lr, decay_steps, reg_change, l1_reg, l2_reg)
        model_eq = get_sympy_expr_v2(model, train_x.shape[0], actual_blocks, round_digits)
        model_complexity = eq_complexity(model_eq)
        model_err = mse_cur + model_complexity * complexity_weight
        model_eq_list.append(model_eq)
        if model_err < best_err:
            best_err = model_err
            best_mse = mse_cur
            best_model = model
            best_train_hist = train_history
            best_blk = actual_blocks
            best_eq = model_eq
    print(model_eq_list)
    return best_model, best_train_hist, best_blk, best_eq, best_mse
