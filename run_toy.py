import os
import numpy as np
import matplotlib as mpl
#mpl.use('Agg') 
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import f1_score, roc_auc_score
from joblib import Parallel, delayed
from scipy.stats import truncnorm
import outlier_gflasso


np.random.seed(1234)
m = 50 # series
n = 500 # time steps
d = 3 # features
p = 5 # steps
o = 10 # outliers
powers_lmbd = list(range(-10, 14, 2))
powers_mu = list(range(-10, 14, 2))
powers_thres = list(range(-10, 14, 1))


def plot_series(X, Y, S, lmbd, mu, p, img_path):
    fig, axs = plt.subplots(X.shape[1], sharex=True)
    title = 'Outlier '
    if p == 2:
        title += 'Group '
    title += 'Fused LASSO Denoising\n' + r'$\lambda={{{}}}, \mu={{{}}}$'.format(lmbd, mu)
    fig.suptitle(title)
    ylabel = 'Features [Dimensionless]'
    for i in range(X.shape[1]):
        axs[i].plot(X[:, i], 'b-', label="Input")
        axs[i].plot(Y[:, i], 'r-', label="Output")
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time [Index]")
    plt.ylabel(ylabel)
    axs[0].legend(loc="upper right")
    #plt.show()
    fig.savefig(img_path, dpi=600)
    #fig.clear()
    plt.close(fig)
    del fig, axs
    gc.collect()


def plot_labels(score, pred, true, thres, lmbd, mu, p, img_path):
    fig, ax = plt.subplots(1, sharex=True)
    title = 'Output Gross-but-'
    if p == 2:
        title += 'Group-'
    title += 'Sparse Norm Values\n of Outlier '
    if p == 2:
        title += 'Group '
    title += 'Fused LASSO Denoising, ' + r'$\lambda={{{}}}, \mu={{{}}}$'.format(lmbd, mu)
    fig.suptitle(title)
    ylabel = 'Output Euclidean Norm [Dimensionless]'
    label = "Norm of gross-but-"
    if p == 2:
        label += "group-"
    label += "sparse first derivative"
    ax.plot(np.arange(len(score)), score, 'b-', label=label)
    ax.plot(np.arange(len(score)), thres * np.ones_like(score), 'r--', label="Threshold for predicted change-point label")
    for idx, t in enumerate(np.where(true == 1)[0]):
        if idx == 0:
            ax.plot(t, 0, 'g.', markersize=14, label="Ground truth logged time of change-point")
        else:
            ax.plot(t, 0, 'g.', markersize=14, label='_nolegend_')
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Time [Index]")
    plt.ylabel(ylabel)
    ax.legend(loc="upper right")
    #plt.show()
    fig.savefig(img_path, dpi=600)
    #fig.clear()
    plt.close(fig)
    del fig, ax
    gc.collect()


if __name__ == "__main__":
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_dir = os.path.join(cwd, 'results/toy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    roc_auc = {}
    f1 = {}
    for q in range(m):
        print(q)
        base_file_name = ('series_' + str(q))

        x = np.arange(n)
        idx = np.random.choice(n - 1, p - 1)
        diff = np.zeros((n - 1, d))
        diff[idx, :] = np.random.randn(p - 1, d)
        init = np.random.randn(d)
        X = np.concatenate([init[None, :], init[None, :] + np.cumsum(diff, axis=0)], axis=0)
        X = np.cumsum(X, axis=0)
        outlier_idx = np.random.choice(n, o)
        X[outlier_idx, :] += 3 * np.random.randn(o, d)
        true = np.zeros((n,))
        true[outlier_idx] = 1
        assert(X.shape[0] == true.shape[0])
        print(true.shape, X.shape)

        def run(lmbd):
            roc_auc_lmbd = {}
            f1_lmbd = {}
            for power_mu in powers_mu:
                mu = 2**power_mu
                for p in [1, 2]:
                    full_file_name = base_file_name + '_res_' + str(lmbd) + '_' + str(mu) + '_' + str(p) 
                    Y_path = os.path.join(save_dir, full_file_name + '_Y.npy')
                    S_path = os.path.join(save_dir, full_file_name + '_S.npy')
                    score_path = os.path.join(save_dir, full_file_name + '_score.npy')
                    if not os.path.exists(Y_path) or not os.path.exists(S_path): # continue from last checkpoint
                        Y, S = outlier_gflasso.run(X=X, lmbd=lmbd, mu=mu, norm_type=p)
                        np.save(Y_path, Y)
                        np.save(S_path, S)
                    else:
                        Y = np.load(Y_path)
                        S = np.load(S_path)
                    if not os.path.exists(score_path):
                        score = np.linalg.norm(S, ord=p, axis=1)
                        np.save(score_path, score)
                    else:                        
                        score = np.load(score_path)
                    roc_auc_lmbd[(lmbd, mu, p)] = roc_auc_score(true, score)
                    
                    img_path = os.path.join(save_dir, full_file_name  + '_series.pdf')
                    if not os.path.exists(img_path): # continue from last checkpoint
                        plot_series(X, Y, S, lmbd, mu, p=p, img_path=img_path)

                    for power_thres in powers_thres:
                        thres = 2**power_thres
                        pred_path = os.path.join(save_dir, full_file_name + '_' + str(thres) + '_pred.npy')
                        pred = (score >= thres)
                        np.save(pred_path, pred)
                        f1_lmbd[(lmbd, mu, p, thres)] = f1_score(true, pred)
                        
                        img_path3 = os.path.join(save_dir, full_file_name  + '_' + str(thres) + '_labels.pdf')
                        if not os.path.exists(img_path3): # continue from last checkpoint
                            plot_labels(score, pred, true, thres, lmbd, mu, p=p, img_path=img_path3)
            return roc_auc_lmbd, f1_lmbd

        results = Parallel(n_jobs=14)(delayed(run)(lmbd=2**power_lmbd) for power_lmbd in powers_lmbd)
        for (power_lmbd, (roc_auc_lmbd, f1_lmbd)) in zip(powers_lmbd, results):
            lmbd = 2**power_lmbd
            for power_mu in powers_mu:
                mu = 2**power_mu
                for p in [1, 2]:
                    print('    lambda=', lmbd, ', mu=', mu, ', p=', p)
                    print('        ROC AUC=', roc_auc_lmbd[(lmbd, mu, p)])
                    for power_thres in powers_thres:
                        thres = 2**power_thres
                        print('        thres=', thres, ', F1=', f1_lmbd[(lmbd, mu, p, thres)])
            for (k, v) in roc_auc_lmbd.items():
                if k not in list(roc_auc.keys()):
                    roc_auc[k] = []
                roc_auc[k].append(v)
            for (k, v) in f1_lmbd.items():
                if k not in list(f1.keys()):
                    f1[k] = []
                f1[k].append(v)



    for p in [1, 2]:
        print(p)

        k_max_roc_auc = None
        v_max_roc_auc = None
        for (k, v) in roc_auc.items():
            if k[2] == p:
                if k_max_roc_auc is None and v_max_roc_auc is None:
                    k_max_roc_auc = k
                    v_max_roc_auc = np.mean(v)
                else:
                    if np.mean(v) > v_max_roc_auc:
                        k_max_roc_auc = k
                        v_max_roc_auc = np.mean(v)
        print('    Best mean ROC AUC score:')
        print('        lambda=', k_max_roc_auc[0], ', mu=', k_max_roc_auc[1], ' : ', v_max_roc_auc, '+-', np.std(roc_auc[k_max_roc_auc]),
                       '(F1 score: ', np.max([np.mean(f1[k_max_roc_auc + (2**power_thres,)]) for power_thres in powers_thres]), 
                       '+-', np.std(f1[k_max_roc_auc + (2**powers_thres[np.argmax([np.mean(f1[k_max_roc_auc + (2**power_thres,)]) for power_thres in powers_thres])],)]), ')')

        k_max_f1 = None
        v_max_f1 = None
        for (k, v) in f1.items():
            if k[2] == p:
                if k_max_f1 is None and v_max_f1 is None:
                    k_max_f1 = k
                    v_max_f1 = np.mean(v)
                else:
                    if np.mean(v) > v_max_f1:
                        k_max_f1 = k
                        v_max_f1 = np.mean(v)
        print('    Best mean F1 score:')
        print('        lambda=', k_max_f1[0], ', mu=', k_max_f1[1], ', thres=', k_max_f1[3], ' : ', v_max_f1, '+-', np.std(f1[k_max_f1]),
                       '(ROC AUC score: ', np.mean(roc_auc[k_max_f1[:3]]), '+-', np.std(roc_auc[k_max_f1[:3]]), ')')

