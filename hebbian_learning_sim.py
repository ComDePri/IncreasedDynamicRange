import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import constants as c
import utils
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
    ASD_IDX = 1

    time_to_90 = np.argmax(Km >= c.LR_THRESHOLD * c.LR_THRESHOLD_PERCENTAGE, axis=1)
    sharp_lr = time_to_90[...,NT_IDX]
    gradual_lr = time_to_90[...,ASD_IDX]

    # Check for differences in learning rate
    wilcoxon_res = scipy.stats.wilcoxon(sharp_lr, gradual_lr, alternative="less")
    print(
        f"Learning rate, mean$\\pm$Std. ASD:${utils.num2print(gradual_lr.mean())}\pm{utils.num2print(gradual_lr.std())}$\n"
        f"NT:${utils.num2print(sharp_lr.mean())}\pm{utils.num2print(sharp_lr.std())}$\n"
        f"Wilcoxon signed-rank test, W({sharp_lr.size - 1})={utils.num2print(wilcoxon_res[0])}, ${utils.num2latex(wilcoxon_res[1])}$")
    lr_fig, subax = pl.plot_learning_rate_and_accuracy(Km, time, None)
    pl.savefig(lr_fig, "learning rate and accuracy", ignore=[subax], shift_x=-0.1, shift_y=1.05, tight=False)

    # Check for differences in bias and variance of the final learned threshold
    last_km_nt = Km[:, -1, NT_IDX]
    last_km_asd = Km[:, -1, ASD_IDX]
    print(
        f"Bias, mean$\\pm$Std. ASD:${utils.num2print(0.5 - last_km_asd.mean())} \\pm {utils.num2print(last_km_asd.std())}$\n"
        f"NT:${utils.num2print(0.5 - last_km_nt.mean())} \\pm {utils.num2print(last_km_nt.std())}$")

    print("Last Km normality test: ")
    nt_normal_stat, nt_normal_p = scipy.stats.shapiro(last_km_nt)
    asd_normal_stat, asd_normal_p = scipy.stats.shapiro(last_km_asd)
    print(f"ASD normality, W={asd_normal_stat}, p<{utils.num2latex(asd_normal_p)}, "
          f"NT normality, W={nt_normal_stat}, p<{utils.num2latex(nt_normal_p)},")
    if nt_normal_p < 0.05 or asd_normal_p < 0.05:
        print("last-learned threshold, Wilcoxon signed-rank test:",
              scipy.stats.wilcoxon(0.5 - last_km_nt, 0.5 - last_km_asd, alternative="less"))
        print("Can't calculate F-test as normality assumption fails!")
    else:
        print("last-learned threshold, paired t-test:",
              scipy.stats.ttest_rel(0.5 - last_km_nt, 0.5 - last_km_asd, alternative="less"))
        var_stat, var_p_val = utils.f_test(last_km_asd, last_km_nt, "greater")
        print(f"Variance equality F-test, ASD var: {last_km_asd.var():.4g}, NT var: {last_km_nt.var():.4g}\n"
              f"F({last_km_asd.size - 1},{last_km_nt.size - 1})={var_stat:.4g}, "
              f"p<{utils.num2latex(var_p_val)}")


if __name__ == '__main__':
    simulate_hebbian_learning()
