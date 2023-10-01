import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import constants as c
import idr_utils as utils
import plotting as pl
from tqdm import tqdm


def simulate_hebbian_learning():
    print("=====================================\n"
          "==== Hebbian learning simulation ====\n"
          "=====================================")

    np.random.seed(c.SEED)
    time = np.arange(0, c.LR_MAX_T + 1, c.LR_DT)
    Km = []
    for _ in tqdm(range(c.LR_N_TRIALS), desc="Hebbian learning trials"):
        Km.append(utils.run_learning_trial())
    Km = np.array(Km)

    NT_IDX = 0
    EI_IDX = 1
    ASD_IDX = 2

    time_to_90 = np.argmax(Km >= c.LR_THRESHOLD * c.LR_THRESHOLD_PERCENTAGE, axis=1)
    nt_lr = time_to_90[..., NT_IDX]
    ei_lr = time_to_90[..., EI_IDX]
    asd_lr = time_to_90[..., ASD_IDX]

    # Check for differences in learning rate
    wilcoxon_res = scipy.stats.wilcoxon(nt_lr, asd_lr, alternative="less")
    print(
        f"ASD-NT:\nLearning rate, mean$\\pm$Std. ASD:${utils.num2print(asd_lr.mean())}\pm{utils.num2print(asd_lr.std())}$\n"
        f"NT:${utils.num2print(nt_lr.mean())}\pm{utils.num2print(nt_lr.std())}$\n"
        f"Wilcoxon signed-rank test, W({nt_lr.size - 1})={utils.num2print(wilcoxon_res[0])}, ${utils.num2latex(wilcoxon_res[1])}$")
    wilcoxon_res = scipy.stats.wilcoxon(ei_lr, asd_lr, alternative="less")
    print("ASD-EI:\n"
          f"Learning rate, mean$\\pm$Std. ASD:${utils.num2print(asd_lr.mean())}\pm{utils.num2print(asd_lr.std())}$\n"
          f"EI:${utils.num2print(ei_lr.mean())}\pm{utils.num2print(ei_lr.std())}$\n"
          f"Wilcoxon signed-rank test, W({ei_lr.size - 1})={utils.num2print(wilcoxon_res[0])}, ${utils.num2latex(wilcoxon_res[1])}$")
    wilcoxon_res = scipy.stats.wilcoxon(nt_lr, ei_lr, alternative="greater")
    print("EI-NT:\n"
          f"Learning rate, mean$\\pm$Std. EI:${utils.num2print(ei_lr.mean())}\pm{utils.num2print(ei_lr.std())}$\n"
          f"NT:${utils.num2print(nt_lr.mean())}\pm{utils.num2print(nt_lr.std())}$\n"
          f"Wilcoxon signed-rank test, W({nt_lr.size - 1})={utils.num2print(wilcoxon_res[0])}, ${utils.num2latex(wilcoxon_res[1])}$")

    utils.reload(pl)
    lr_fig, subax = pl.plot_learning_rate_and_accuracy(Km, time, None)
    pl.savefig(lr_fig, "learning rate and accuracy", ignore=[subax], shift_x=-0.1, shift_y=1.05, tight=False)
    pl.plt.close()
    # Check for differences in bias and variance of the final learned threshold
    last_km_nt = Km[:, -1, NT_IDX]
    last_km_ei = Km[:, -1, EI_IDX]
    last_km_asd = Km[:, -1, ASD_IDX]
    print(
        f"Bias, mean$\\pm$Std. ASD:${utils.num2print(0.5 - last_km_asd.mean())} \\pm {utils.num2print(last_km_asd.std())}$\n"
        f"NT:${utils.num2print(0.5 - last_km_nt.mean())} \\pm {utils.num2print(last_km_nt.std())}$"
        f"EI:${utils.num2print(0.5 - last_km_ei.mean())} \\pm {utils.num2print(last_km_ei.std())}$")

    print("Last Km normality test: ")
    nt_normal_stat, nt_normal_p = scipy.stats.shapiro(last_km_nt)
    asd_normal_stat, asd_normal_p = scipy.stats.shapiro(last_km_asd)
    ei_normal_stat, ei_normal_p = scipy.stats.shapiro(last_km_ei)
    print(f"ASD normality, W={asd_normal_stat}, p<{utils.num2latex(asd_normal_p)}, "
          f"NT normality, W={nt_normal_stat}, p<{utils.num2latex(nt_normal_p)},"
          f"EI normality, W={ei_normal_stat}, p<{utils.num2latex(ei_normal_p)},")
    if nt_normal_p < 0.05 or asd_normal_p < 0.05 and ei_normal_p < 0.05:
        print("NT-ASD: last-learned threshold, Wilcoxon signed-rank test:",
              scipy.stats.wilcoxon(0.5 - last_km_nt, 0.5 - last_km_asd, alternative="less"))
        print("EI-ASD: last-learned threshold, Wilcoxon signed-rank test:",
              scipy.stats.wilcoxon(0.5 - last_km_ei, 0.5 - last_km_asd, alternative="less"))
        print("EI-NT: last-learned threshold, Wilcoxon signed-rank test:",
              scipy.stats.wilcoxon(0.5 - last_km_ei, 0.5 - last_km_nt, alternative="less"))
        print("Can't calculate F-test as normality assumption fails!")
    else:
        print("NT-ASD: last-learned threshold, paired t-test:",
              scipy.stats.ttest_rel(0.5 - last_km_nt, 0.5 - last_km_asd, alternative="less"))
        print("EI-ASD: last-learned threshold, paired t-test:",
              scipy.stats.ttest_rel(0.5 - last_km_ei, 0.5 - last_km_asd, alternative="less"))
        print("EI-NT: last-learned threshold, paired t-test:",
              scipy.stats.ttest_rel(0.5 - last_km_ei, 0.5 - last_km_nt, alternative="less"))

        var_stat, var_p_val = utils.f_test(last_km_asd, last_km_nt, "greater")
        print(f"Variance equality F-test, ASD var: {last_km_asd.var():.4g}, NT var: {last_km_nt.var():.4g}\n"
              f"F({last_km_asd.size - 1},{last_km_nt.size - 1})={var_stat:.4g}, "
              f"p<{utils.num2latex(var_p_val)}")

        var_stat, var_p_val = utils.f_test(last_km_asd, last_km_ei, "greater")
        print(f"Variance equality F-test, ASD var: {last_km_asd.var():.4g}, EI var: {last_km_ei.var():.4g}\n"
              f"F({last_km_asd.size - 1},{last_km_ei.size - 1})={var_stat:.4g}, "
              f"p<{utils.num2latex(var_p_val)}")

        var_stat, var_p_val = utils.f_test(last_km_ei, last_km_nt, "greater")
        print(f"Variance equality F-test, EI var: {last_km_ei.var():.4g}, NT var: {last_km_nt.var():.4g}\n"
              f"F({last_km_asd.size - 1},{last_km_nt.size - 1})={var_stat:.4g}, "
              f"p<{utils.num2latex(var_p_val)}")


if __name__ == '__main__':
    simulate_hebbian_learning()
