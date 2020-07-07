import numpy as np

#Project adv example on l_{\infty} adv_bound ball centered on clean example
def clip_adv(X_adv, X_test, indices_test, adv_bound):
    X_clip_adv = np.zeros(X_test[indices_test].shape)
    j = 0
    for i in indices_test:
        diff = X_adv[j] - X_test[i]
        X_clip_adv[j] = X_test[i] + np.clip(diff, -adv_bound, adv_bound)
        j = j +1
    return(X_clip_adv)
    
#Project adv example on l_2 adv_bound ball centered on clean example
def clip_adv_l2(X_adv, X_test, indices_test, adv_bound):
    X_clip_adv = np.zeros(X_test[indices_test].shape)
    j = 0
    for i in indices_test:
        diff = X_adv[j] - X_test[i]
        norm = np.sqrt(np.maximum(1e-12, np.sum(np.square(diff))))
        # We must *clip* to within the norm ball, not *normalize* onto the# surface of the ball
        factor = np.minimum(1., np.divide(adv_bound, norm))
        diff = diff * factor
        X_clip_adv[j] = X_test[i] + diff
        j = j +1
    return(X_clip_adv)
    
#Function which returns the adversarial accuracy, the number of successful adversarial examples, l2 and l_inf distances between successful adversarial examples and clean observations
def metrics(model, X_adv, X_test, y_pred, indices_test):    
    adv_pred = np.argmax(model.predict(X_adv), axis = 1)
    adv_acc =  np.mean(np.equal(adv_pred, y_pred[indices_test]))
    l2_distort_success = 0
    linf_distort_success = 0
    l1_distort_success = 0
    l0_distort_success = 0
    l2_distort_fail = 0
    linf_distort_fail = 0
    l1_distort_fail = 0
    l0_distort_fail = 0
    nb_success = 0
    j = 0
    for i in indices_test:
        if (adv_pred[j] != y_pred[i]):
            l2_distort_success = l2_distort_success + np.linalg.norm(X_test[i] - X_adv[j])
            linf_distort_success = linf_distort_success + np.max(abs(X_test[i] - X_adv[j]))
            l1_distort_success = l1_distort_success + np.sum(np.abs(X_test[i] - X_adv[j]))
            l0_distort_success = l0_distort_success +  np.linalg.norm(X_test[i].flatten() - X_adv[j].flatten(), ord=0)
            nb_success = nb_success + 1     
        if (adv_pred[j] == y_pred[i]):
            l2_distort_fail = l2_distort_fail + np.linalg.norm(X_test[i] - X_adv[j])
            linf_distort_fail = linf_distort_fail + np.max(abs(X_test[i] - X_adv[j]))
            l1_distort_fail = l1_distort_fail + np.sum(np.abs(X_test[i] - X_adv[j]))
            l0_distort_fail = l0_distort_fail +  np.linalg.norm(X_test[i].flatten() - X_adv[j].flatten(), ord=0)
        j = j+1        
    nb_fail = len(indices_test) - nb_success
    if ((nb_fail != 0) & (nb_success != 0)):
        return(adv_acc, nb_success, l2_distort_success/nb_success, linf_distort_success/nb_success, l1_distort_success/nb_success,
               l0_distort_success/nb_success, l2_distort_fail/nb_fail, linf_distort_fail/nb_fail, l1_distort_fail/nb_fail,
               l0_distort_fail/nb_fail)
    elif (nb_fail == 0):
        return(adv_acc, nb_success, l2_distort_success/nb_success, linf_distort_success/nb_success, l1_distort_success/nb_success,
               l0_distort_success/nb_success, "non", "non", "non", "non")
    elif (nb_success == 0):
        return(adv_acc, nb_success, "non", "non", "non", "non", l2_distort_fail/nb_fail, linf_distort_fail/nb_fail, l1_distort_fail/nb_fail,
               l0_distort_fail/nb_fail)
        
def agree_func(indices_test, pred_adv, pred_adv_tot, pred, pred_tot):    
    
    c_1 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] != pred_adv_tot[i]):
            if (pred_adv[i] == pred[indices_test[i]]):
                c_1 = c_1+1 
    print("Detected and well-classified by base: " + str(c_1))
    
    c_2 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] != pred_adv_tot[i]):
            if (pred_adv[i] != pred[indices_test[i]]):
                c_2 = c_2+1 
    print("Detected and badly-classified by base: " + str(c_2))
    
    c_3 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] == pred_adv_tot[i]):
            if (pred_adv[i] == pred[indices_test[i]]):
                c_3 = c_3+1 
    print("Undetected and well-classified by base: " + str(c_3))
    
    c_4 = 0.0
    for i in range(len(indices_test)):
        if (pred_adv[i] == pred_adv_tot[i]):
            if (pred_adv[i] != pred[indices_test[i]]):
                c_4 = c_4+1 
    print("Undetected and badly-classified by base: " + str(c_4))
    
    print((c_1 + c_3) / len(indices_test))
    print((c_1 + c_2 + c_3) / len(indices_test))

def comp_func(X_adv_stacked, X_adv_auto, X_adv_ce, X_adv_rob, indices_test, pred_base, pred_stacked, pred_auto, pred_ce, pred_rob):
    
    pred_stacked_adv = np.argmax(model_stacked.predict(X_adv_stacked), axis = 1)
    pred_auto_adv = np.argmax(model_auto.predict(X_adv_auto), axis = 1)
    pred_ce_adv = np.argmax(model_ce.predict(X_adv_ce), axis = 1)
    pred_rob_adv = np.argmax(model_rob.predict(X_adv_rob), axis = 1)
    
    success_indices_stacked_adv = np.not_equal(pred_stacked_adv, y_test[indices_test])
    success_indices_auto_adv = np.not_equal(pred_auto_adv, y_test[indices_test])
    success_indices_ce_adv = np.not_equal(pred_ce_adv, y_test[indices_test])
    success_indices_rob_adv = np.not_equal(pred_rob_adv, y_test[indices_test])
    
    print(np.sum(success_indices_stacked_adv))
    print(np.sum(success_indices_auto_adv))
    print(np.sum(success_indices_ce_adv))
    print(np.sum(success_indices_rob_adv))
    
    cond = (success_indices_stacked_adv == success_indices_auto_adv) & (success_indices_auto_adv == success_indices_ce_adv) & (success_indices_ce_adv == success_indices_rob_adv) & (success_indices_rob_adv == True) 
    success_indices_adv = indices_test[cond]
    
    print("metrics source models")
    print(metrics(model_stacked, X_adv_stacked[cond], X_test, pred_stacked, success_indices_adv))
    print(metrics(model_auto, X_adv_auto[cond], X_test, pred_auto, success_indices_adv))
    print(metrics(model_ce, X_adv_ce[cond], X_test, pred_ce, success_indices_adv))    
    print(metrics(model_rob, X_adv_rob[cond], X_test, pred_rob, success_indices_adv))
    
    print("metrics base model")
    print(metrics(model, X_adv_stacked[cond], X_test, pred_base, success_indices_adv))
    print(metrics(model, X_adv_auto[cond], X_test, pred_base, success_indices_adv))
    print(metrics(model, X_adv_ce[cond], X_test, pred_base, success_indices_adv))
    print(metrics(model, X_adv_rob[cond], X_test, pred_base, success_indices_adv))


    pred_adv_basefromstacked = np.argmax(model.predict(X_adv_stacked[cond]), axis=1)
    pred_adv_basefromauto = np.argmax(model.predict(X_adv_auto[cond]), axis=1)
    pred_adv_basefromce = np.argmax(model.predict(X_adv_ce[cond]), axis=1)
    pred_adv_basefromrob = np.argmax(model.predict(X_adv_rob[cond]), axis=1)

    agree_func(success_indices_adv, pred_adv_basefromstacked, pred_stacked_adv[cond], pred_base, pred_stacked)
    agree_func(success_indices_adv, pred_adv_basefromauto, pred_auto_adv[cond], pred_base, pred_auto)
    agree_func(success_indices_adv, pred_adv_basefromce, pred_ce_adv[cond], pred_base, pred_ce)
    agree_func(success_indices_adv, pred_adv_basefromrob, pred_rob_adv[cond], pred_base, pred_rob) 

def comp_func_transfer(X_adv_source, indices_test, pred_base, pred_source, model_source, model_base):
    
    print("metrics source model")
    print(metrics(model_source, X_adv_source, X_test, pred_source, indices_test))
    print("metrics base model")
    print(metrics(model_base, X_adv_source, X_test, pred_base, indices_test))

    pred_source_adv = np.argmax(model_source.predict(X_adv_source), axis = 1)
    pred_adv_basefromsource = np.argmax(model_base.predict(X_adv_source), axis=1)  
    agree_func(indices_test, pred_adv_basefromsource, pred_source_adv, pred_base, pred_source)
